import torch
import numpy as np
import os
import json
import pandas as pd
from models.SimCLR import SimCLR
from dataset import data
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.preprocessing import normalize
import faiss

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

def evaluate_group_retrieval(features, image_names, index, groups, k=5):
    #We get the file name without extension
    image_names = normalize_path(image_names)

    name_to_idx = {name: idx for idx, name in enumerate(image_names)}
    total_correct = 0
    total_groups = 0

    for group in groups:
        # Filter to only images present in the index
        
        filtered_group = [name for name in group if name in name_to_idx]

        group_indices = set(name_to_idx[name] for name in filtered_group)
        
        for img_name in filtered_group:
            idx = name_to_idx[img_name]
            query_vec = features[idx].reshape(1, -1)

            # Search for k nearest neighbors (including itself)
            distances, neighbors = index.search(query_vec, k)
            neighbors = neighbors[0]  # neighbors array
            # Count neighbors (excluding itself) that are in the same group
            same_group_count = sum((neighbor in group_indices and neighbor != idx) for neighbor in neighbors)

            if same_group_count > 0:
                total_correct += 1
            total_groups += 1

    accuracy = total_correct / total_groups if total_groups > 0 else 0.0
    print(f"Group retrieval accuracy (at least 1 neighbor in group among top {k}): {accuracy*100:.2f}%")
    return accuracy

def recall_at_k(index, embeddings, paths, groups, k=5):
    """
    Calcule le Recall@K sur les groupes d'images similaires (near duplicates).
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    total = 0
    hits = 0

    for group in groups:
        group_set = set(group)
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]
            hit = any(p in group_set for p in retrieved_paths)
            hits += int(hit)
            total += 1

    return hits / total if total > 0 else 0.0

def precision_at_k(index, embeddings, paths, groups, k=5):
    """
    Calcule la Precision@K sur les groupes d'images similaires.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}
    
    total = 0
    precision_sum = 0.0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path])
            
            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]
            
            if not retrieved_paths:
                continue
            
            relevant_count = sum(1 for p in retrieved_paths if p in query_group)
            precision = relevant_count / len(retrieved_paths)
            precision_sum += precision
            total += 1

    return precision_sum / total if total > 0 else 0.0

def mean_average_precision(index, embeddings, paths, groups, k=10):
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    average_precisions = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]

            relevant_hits = 0
            precision_sum = 0.0

            for rank, retrieved_path in enumerate(retrieved_paths, start=1):
                if retrieved_path in query_group:
                    relevant_hits += 1
                    precision_sum += relevant_hits / rank

            if relevant_hits > 0:
                average_precisions.append(precision_sum / len(query_group))
            else:
                average_precisions.append(0.0)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

def f1_score_at_k(index, embeddings, paths, groups, k=5):
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    f1_scores = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]

            if not retrieved_paths:
                continue

            true_positives = sum(1 for p in retrieved_paths if p in query_group)
            precision = true_positives / len(retrieved_paths)
            recall = true_positives / len(query_group) if len(query_group) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def load_groundtruth(gt_path):
    df = pd.read_excel(gt_path, header=None)
    groups = []
    # Iterate through all cells and collect non-null, cleaned filenames
    for row in df.itertuples(index=False):
        current_groups = []
        for cell in row:
            if pd.notnull(cell):
                filename = str(cell).strip().lower().split('.')[0]  # remove extension
                current_groups.append(filename)
        groups.append(current_groups)
    return groups

# ------------------- Main -------------------
# Load model
backbone = resnet50(pretrained=False)
model = SimCLR(backbone)
'''
checkpoint = torch.load('/Users/nour/Desktop/MSV/models/simclr_model_50.pth', map_location='mps')  
model.load_state_dict(checkpoint)
model.eval()
'''
# Prepare data
image_directory = '/Users/nour/Desktop/historicaldataset'

groups = load_groundtruth('/Users/nour/Desktop/groundtruth.xlsx')
groups = [group for group in groups if len(group) >= 2]
absolute_paths = [find_image_path(image_directory, name) for group in groups for name in group]
absolute_paths = [path for path in absolute_paths if path is not None]

# Extract features and build index
features, image_names, index = extract_features_and_build_index(model, absolute_paths)
# Save features and index if you want

np.savez('cbir_features.npz', features=features, image_names=image_names)
faiss.write_index(index, 'cbir_faiss.index')

# Evaluate retrieval
accuracy = evaluate_group_retrieval(features, image_names, index, groups, k=5)
'''
image_names = normalize_path(image_names)

# Évaluation Recall@K
recall_k1 = recall_at_k(index, features, image_names, groups, k=1)
recall_k5 = recall_at_k(index, features, image_names, groups, k=5)
recall_k10 = recall_at_k(index, features, image_names, groups, k=10)

# Évaluation Precision@K
precision_k1 = precision_at_k(index, features, image_names, groups, k=1)
precision_k5 = precision_at_k(index, features, image_names, groups, k=5)
precision_k10 = precision_at_k(index, features, image_names, groups, k=10)

map_k10 = mean_average_precision(index, features, image_names, groups, k=10)
f1_k5 = f1_score_at_k(index, features, image_names, groups, k=5)

results = {
    "recall@1": round(recall_k1, 4),
    "recall@5": round(recall_k5, 4),
    "recall@10": round(recall_k10, 4),
    "precision@1": round(precision_k1, 4),
    "precision@5": round(precision_k5, 4),
    "precision@10": round(precision_k10, 4),
    "mAP@10": round(map_k10, 4),
    "f1@5": round(f1_k5, 4),
}

print("\n✅ Évaluation terminée et résultats sauvegardés.")
print(json.dumps(results, indent=4))
'''