import torch
import os
import time
import json
import pandas as pd
import numpy as np
import faiss
import argparse
from models.SimCLR import SimCLR
from models.FineTunedDino import FineTunedDino
from dataset import data
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.preprocessing import normalize
from utils.metrics import f1_score_at_k, recall_at_k, precision_at_k, mean_average_precision, evaluate_with_distance_threshold, evaluate_group_retrieval
from utils.metrics import f1_score_at_k_no_faiss, recall_at_k_no_faiss, precision_at_k_no_faiss, mean_average_precision_no_faiss, compute_top_k_neighbors

def find_image_path(image_dir, name):
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif']:
        full_path = os.path.join(image_dir, name + ext)
        if os.path.exists(full_path):
            return full_path
    return None

def extract_features_and_build_index(model, image_paths, batch_size=32):
    model.eval()
    representation = []
    image_names = []

    test_data = data.CBIRImageDataset(image_paths)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_imgs, batch_path in test_loader:
            h, _ = model(batch_imgs)
            representation.append(h.cpu())
            image_names.extend(batch_path)

    features = torch.cat(representation, dim=0).numpy().astype('float32')
    features = normalize(features, axis=1)  # Normalize for cosine similarity

    dimension = features.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(features)
    
    assert features.shape[0] == len(image_names), "Final mismatch between features and image names"

    return features, image_names, index

def normalize_path(image_names):
    images = []
    for img in image_names:
        base_name = os.path.splitext(os.path.basename(img))[0]
        images.append(base_name)
    return images

def load_groundtruth(gt_path):
    df = pd.read_excel(gt_path, header=None)
    groups = []
    for row in df.itertuples(index=False):
        current_groups = []
        for cell in row:
            if pd.notnull(cell):
                filename = str(cell).strip().lower().split('.')[0]
                current_groups.append(filename)
        groups.append(current_groups)
    return groups

def evaluate_faiss(img_path, ground_truth_path, model='resnet50', checkpoint_path='', save=False, dim=64):
    
    if(model == 'simclr'):
        backbone = resnet50(pretrained=False)
        model = SimCLR(backbone, dim)
        checkpoint = torch.load(checkpoint_path, map_location='mps')
        state_dict = {
        (k.replace('module.', '', 1) if k.startswith('module.') else k): v
        for k, v in checkpoint.items()
        }
        model.load_state_dict(state_dict)
    elif(model == 'dino'):
        model = FineTunedDino()
        checkpoint = torch.load(checkpoint_path, map_location='mps')
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
    elif(model == 'resnet50'):
        backbone = resnet50(pretrained=True)
        model = SimCLR(backbone)
        
    model.eval()

    groups = load_groundtruth(ground_truth_path)
    groups = [group for group in groups if len(group) >= 2]
    absolute_paths = [find_image_path(img_path, name) for group in groups for name in group]
    absolute_paths = [path for path in absolute_paths if path is not None]

    features, image_names, index = extract_features_and_build_index(model, absolute_paths)
    
    if(save):
        np.savez('cbir_features.npz', features=features, image_names=image_names)
        faiss.write_index(index, 'cbir_faiss.index')

    image_names = normalize_path(image_names)

    #accuracy = evaluate_group_retrieval(features, image_names, index, groups, k=5)
    
    start_time = time.time()

    recall_k1 = recall_at_k(index, features, image_names, groups, k=1)
    recall_k5 = recall_at_k(index, features, image_names, groups, k=5)
    recall_k10 = recall_at_k(index, features, image_names, groups, k=10)

    precision_k1 = precision_at_k(index, features, image_names, groups, k=1)
    precision_k5 = precision_at_k(index, features, image_names, groups, k=5)
    precision_k10 = precision_at_k(index, features, image_names, groups, k=10)

    map_k5 = mean_average_precision(index,features,image_names, groups, k=5)
    map_k10 = mean_average_precision(index, features, image_names, groups, k=10)

    f1_k5 = f1_score_at_k(index, features, image_names, groups, k=5)
    f1_k10 = f1_score_at_k(index, features, image_names, groups, k=10)

    threshold, precision, recall, f1 = evaluate_with_distance_threshold(index, features, image_names, groups, k=10, threshold_type="mean")
    
    end_time = time.time() - start_time
    
    results = {
        "recall@1": round(recall_k1, 4),
        "recall@5": round(recall_k5, 4),
        "recall@10": round(recall_k10, 4),
        "precision@1": round(precision_k1, 4),
        "precision@5": round(precision_k5, 4),
        "precision@10": round(precision_k10, 4),
        "mAP@5": round(map_k5, 4),
        "mAP@10": round(map_k10, 4),
        "f1@5": round(f1_k5, 4),

        "f1@10": round(f1_k10, 4),
        "threshold": round(threshold, 4),
        "precision_thresh": round(precision, 4),
        "recall_thresh": round(recall, 4),
        "f1_thresh": round(f1, 4),
        "compute_time": end_time
    }

    print("\n Évaluation terminée et résultats sauvegardés.")
    print(json.dumps(results, indent=4))

def evaluate_with_distance_no_faiss(img_path, ground_truth_path, model='resnet50', checkpoint_path='', save=False, dim=64, k=10, threshold_type='mean'):
    """
    Évalue la performance du système de recherche d'images par similarité en utilisant un seuil sur la distance,
    sans FAISS (utilise uniquement numpy pour les calculs).

    Args:
        embeddings (np.ndarray): Embeddings des images (shape: [N, D]).
        paths (List[str]): Chemins des images correspondant aux embeddings.
        groups (List[List[Dict]]): Groupes d'images similaires (ground truth) sous forme [{image, caption}, ...].
        k (int): Nombre de voisins à rechercher pour chaque image (k+1 en réalité car on ignore l'image elle-même).
        threshold_type (str): Méthode pour calculer le seuil de distance. Options : "mean", "median", "percentile_25".

    Returns:
        threshold (float): Seuil utilisé pour filtrer les voisins.
        precision (float): Précision globale (TP / (TP + FP)).
        recall (float): Rappel global (TP / (TP + FN)).
        f1 (float): Score F1 global.
    """
    if(model == 'simclr'):
        backbone = resnet50(pretrained=False)
        model = SimCLR(backbone, dim)
        checkpoint = torch.load(checkpoint_path, map_location='mps')
        state_dict = {
        (k.replace('module.', '', 1) if k.startswith('module.') else k): v
        for k, v in checkpoint.items()
        }
        model.load_state_dict(state_dict)
    elif(model == 'dino'):
        model = FineTunedDino()
        checkpoint = torch.load(checkpoint_path, map_location='mps')
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
    elif(model == 'resnet50'):
        backbone = resnet50(pretrained=True)
        model = SimCLR(backbone)
        
    model.eval()

    groups = load_groundtruth(ground_truth_path)
    groups = [group for group in groups if len(group) >= 2]
    absolute_paths = [find_image_path(img_path, name) for group in groups for name in group]
    absolute_paths = [path for path in absolute_paths if path is not None]

    features, image_names, index = extract_features_and_build_index(model, absolute_paths)

    if(save):
        np.savez('cbir_features.npz', features=features, image_names=image_names)
        faiss.write_index(index, 'cbir_faiss.index')

    image_names = normalize_path(image_names)

    groups_image_names = [[img['image'] if isinstance(img, dict) else img for img in group] for group in groups]
    groups_image_names = [[normalize_path([img])[0] for img in group] for group in groups_image_names]

    path_to_index = {path: i for i, path in enumerate(image_names)}
    
    start_no_faiss = time.time()

    # Collecter les distances pour calculer le seuil global
    all_distances = []

    for i in range(len(features)):
        query_embedding = features[i]
        distances = np.linalg.norm(features - query_embedding, axis=1)
        distances[i] = np.inf  # ignore self-distance
        all_distances.extend(distances[np.isfinite(distances)])

    all_distances = np.array(all_distances)

    # Calcul du seuil
    if threshold_type == "mean":
        threshold = np.mean(all_distances)
    elif threshold_type == "median":
        threshold = np.median(all_distances)
    elif threshold_type == "percentile_25":
        threshold = np.percentile(all_distances, 25)
    else:
        raise ValueError(f"Unsupported threshold_type: {threshold_type}")
    
    TP = 0
    FP = 0
    FN = 0

    for group in groups_image_names:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_embedding = features[query_idx]
            query_group = set(group) - {query_path}

            distances = np.linalg.norm(features - query_embedding, axis=1)
            distances[query_idx] = np.inf

            neighbor_indices = np.argsort(distances)[:k]
            retrieved_paths = [
                image_names[i]
                for i in neighbor_indices
                if distances[i] < threshold and image_names[i] != query_path
            ]
            retrieved_set = set(retrieved_paths)

            tp = len(retrieved_set & query_group)
            fp = len(retrieved_set - query_group)
            fn = len(query_group - retrieved_set)

            TP += tp
            FP += fp
            FN += fn
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    recall_k1_no_faiss = recall_at_k_no_faiss(features, image_names, groups, k=1)
    recall_k5_no_faiss = recall_at_k_no_faiss(features, image_names, groups, k=5)
    recall_k10_no_faiss = recall_at_k_no_faiss(features, image_names, groups, k=10)

    precision_k1_no_faiss = precision_at_k_no_faiss(features, image_names, groups, k=1)
    precision_k5_no_faiss = precision_at_k_no_faiss(features, image_names, groups, k=5)
    precision_k10_no_faiss = precision_at_k_no_faiss(features, image_names, groups, k=10)

    map_k5_no_faiss = mean_average_precision_no_faiss(features, image_names, groups, k=5)
    map_k10_no_faiss = mean_average_precision_no_faiss(features, image_names, groups, k=10)

    f1_k5_no_faiss = f1_score_at_k_no_faiss(features, image_names, groups, k=5)
    f1_k10_no_faiss = f1_score_at_k_no_faiss(features, image_names, groups, k=10)


    no_faiss_time = time.time() - start_no_faiss

    results = {
            "recall@1": round(recall_k1_no_faiss, 4),
            "recall@5": round(recall_k5_no_faiss, 4),
            "recall@10": round(recall_k10_no_faiss, 4),
            "precision@1": round(precision_k1_no_faiss, 4),
            "precision@5": round(precision_k5_no_faiss, 4),
            "precision@10": round(precision_k10_no_faiss, 4),
            "mAP@5": round(map_k5_no_faiss, 4),
            "mAP@10": round(map_k10_no_faiss, 4),
            "f1@5": round(f1_k5_no_faiss, 4),

            "f1@10": round(f1_k10_no_faiss, 4),
            "threshold": round(threshold, 4),
            "precision_thresh": round(precision, 4),
            "recall_thresh": round(recall, 4),
            "f1_thresh": round(f1, 4),
            "compute time": round(no_faiss_time)
        }
    
    # Convert all numpy float32 values to native Python floats
    results = {k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v for k, v in results.items()}
    print("\n Évaluation terminée pour no Faiss et résultats sauvegardés.")
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    valid_models = ['resnet50', 'simclr', 'dino']

    parser.add_argument('--image_path','-ip', type=str, required=True)
    parser.add_argument('--gt_path', '-gt', type=str, required=True)
    parser.add_argument(
        '--model',
        type=str,
        choices=valid_models,
        help=f"Mode must be one of: {', '.join(valid_models)}",
        default='resnet50'
        )    
    parser.add_argument('--checkpoint', '-chk', type=str)
    parser.add_argument('--save', type=bool)
    parser.add_argument('--dim', type=int, default = 64)
    args = parser.parse_args()

    evaluate_faiss(args.image_path, args.gt_path, args.model, args.checkpoint, args.save, args.dim)
    evaluate_with_distance_no_faiss(args.image_path, args.gt_path, args.model, args.checkpoint, args.save, args.dim)