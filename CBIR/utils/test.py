import torch
import os
import json
import pandas as pd
from models.SimCLR import SimCLR
from dataset import data
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.preprocessing import normalize
import faiss
from utils.metrics import f1_score_at_k, recall_at_k, precision_at_k, mean_average_precision, evaluate_with_distance_threshold, evaluate_group_retrieval

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


backbone = resnet50(pretrained=False)
model = SimCLR(backbone)


checkpoint = torch.load('/Users/nour/Desktop/MSV/CBIR/runs/run_4/simclr_model_epoch_20.pth', map_location='mps')
model.load_state_dict(checkpoint)

model.eval()

image_directory = '/Users/nour/Desktop/historicaldataset'

groups = load_groundtruth('/Users/nour/Desktop/groundtruth.xlsx')
groups = [group for group in groups if len(group) >= 2]
absolute_paths = [find_image_path(image_directory, name) for group in groups for name in group]
absolute_paths = [path for path in absolute_paths if path is not None]

features, image_names, index = extract_features_and_build_index(model, absolute_paths)

#np.savez('cbir_features.npz', features=features, image_names=image_names)
#faiss.write_index(index, 'cbir_faiss.index')


image_names = normalize_path(image_names)

accuracy = evaluate_group_retrieval(features, image_names, index, groups, k=5)

recall_k1 = recall_at_k(index, features, image_names, groups, k=1)
recall_k5 = recall_at_k(index, features, image_names, groups, k=5)
recall_k10 = recall_at_k(index, features, image_names, groups, k=10)

precision_k1 = precision_at_k(index, features, image_names, groups, k=1)
precision_k5 = precision_at_k(index, features, image_names, groups, k=5)
precision_k10 = precision_at_k(index, features, image_names, groups, k=10)

map_k5 = mean_average_precision(index,features,image_names, groups, k=5)
map_k10 = mean_average_precision(index, features, image_names, groups, k=10)

f1_k2 = f1_score_at_k(index, features, image_names, groups, k=2)
f1_k5 = f1_score_at_k(index, features, image_names, groups, k=5)
f1_k10 = f1_score_at_k(index, features, image_names, groups, k=10)

threshold, precision, recall, f1 = evaluate_with_distance_threshold(index, features, image_names, groups, k=10, threshold_type="mean")

results = {
    "recall@1": round(recall_k1, 4),
    "recall@5": round(recall_k5, 4),
    "recall@10": round(recall_k10, 4),
    "precision@1": round(precision_k1, 4),
    "precision@5": round(precision_k5, 4),
    "precision@10": round(precision_k10, 4),
    "mAP@5": round(map_k5, 4),
    "mAP@10": round(map_k10, 4),
    "f1@2": round(f1_k2, 4),
    "f1@5": round(f1_k5, 4),

    "f1@10": round(f1_k10, 4),
    "threshold": round(threshold, 4),
    "precision_thresh": round(precision, 4),
    "recall_thresh": round(recall, 4),
    "f1_thresh": round(f1, 4)
}

print("\n Évaluation terminée et résultats sauvegardés.")
print(json.dumps(results, indent=4))
