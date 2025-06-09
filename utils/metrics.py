import numpy as np


def evaluate_group_retrieval(features, image_names, index, groups, k=5):
    name_to_idx = {name: idx for idx, name in enumerate(image_names)}
    total_correct = 0
    total_groups = 0

    for group in groups:
        filtered_group = [name for name in group if name in name_to_idx]

        group_indices = set(name_to_idx[name] for name in filtered_group)
        
        for img_name in filtered_group:
            idx = name_to_idx[img_name]
            query_vec = features[idx].reshape(1, -1)

            distances, neighbors = index.search(query_vec, k)
            neighbors = neighbors[0]
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
    total_recall = 0.0
    valid_queries = 0

    for group in groups:
        group_set = set(group)
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path]
            retrieved_set = set(retrieved_paths)
            gt_set = group_set - {query_path}
            if not gt_set:
                print('we are skipping things here...')
                continue

            correct_retrieved = len(retrieved_set & gt_set)
            recall = correct_retrieved / len(gt_set)

            total_recall += recall
            valid_queries += 1

    return total_recall / valid_queries if valid_queries > 0 else 0.0

def precision_at_k(index, embeddings, paths, groups, k=5):
    """
    Compute Precision@K for groups of similar images (near duplicates).
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_queries = 0
    total_precision = 0.0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path][:k]

            # Count relevant retrieved images
            relevant_count = sum(p in query_group for p in retrieved_paths)

            precision = relevant_count / k
            total_precision += precision
            total_queries += 1

    return total_precision / total_queries if total_queries > 0 else 0.0

def mean_average_precision(index, embeddings, paths, groups, k=10):
    """
    Compute mean Average Precision (mAP@K) for a set of image groups.
    Each image is treated as a query, and the rest of its group is its ground truth.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}
    path_set = set(paths)

    average_precisions = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}
            query_group = query_group & path_set

            if not query_group:
                continue

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path][:k]

            relevant_hits = 0
            precision_sum = 0.0

            for rank, retrieved_path in enumerate(retrieved_paths, start=1):
                if retrieved_path in query_group:
                    relevant_hits += 1
                    precision_sum += relevant_hits / rank

            ap = precision_sum / len(query_group)
            average_precisions.append(ap)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def f1_score_at_k(index, embeddings, paths, groups, k=5):
    """
    Compute F1-score@K over groups of similar images (near duplicates).
    Each image in a group is treated as a query, and the rest of its group is ground truth.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    f1_scores = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}

            if not query_group:
                continue

            _, I = index.search(np.array([embeddings[query_idx]]).astype(np.float32), k + 1)
            retrieved_paths = [paths[i] for i in I[0] if paths[i] != query_path][:k]

            true_positives = sum(1 for p in retrieved_paths if p in query_group)

            precision = true_positives / k
            recall = true_positives / len(query_group)

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def evaluate_with_distance_threshold(index, embeddings, paths, groups, k=10, threshold_type="mean"):
    """
    Évalue la performance du système de recherche d'images par similarité en utilisant un seuil sur la distance
    pour filtrer les voisins.

    Paramètres :
        index (faiss.Index): Index FAISS pour la recherche.
        embeddings (np.ndarray): Embeddings des images.
        paths (List[str]): Chemins des images correspondant aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins à rechercher pour chaque image (k+1 en réalité car on ignore l'image elle-même).
        threshold_type (str): Méthode pour calculer le seuil de distance. 
            Options : "mean", "median", "percentile_25".

    Retourne :
        threshold (float): Seuil utilisé pour filtrer les voisins.
        precision (float): Précision globale (TP / (TP + FP)).
        recall (float): Rappel global (TP / (TP + FN)).
        f1 (float): Score F1 global.
    """
    
    # Création d'un dictionnaire pour accéder rapidement à l'index et au groupe de chaque image
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    # Recherche des k+1 plus proches voisins pour chaque image
    D, _ = index.search(embeddings.astype(np.float32), k + 1)

    # Calcul du seuil de distance selon la stratégie choisie
    if threshold_type == "mean":
        threshold = np.mean(D[:, 1:])  # on ignore la première colonne (distance à soi-même)
    elif threshold_type == "median":
        threshold = np.median(D[:, 1:])
    elif threshold_type == "percentile_25":
        threshold = np.percentile(D[:, 1:], 25)
    else:
        raise ValueError("Unsupported threshold_type")

    # Initialisation des compteurs pour TP, FP, FN
    TP = 0
    FP = 0
    FN = 0

    # Boucle sur chaque image de chaque groupe
    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue  # sécurité si le chemin est manquant

            query_idx = path_to_index[query_path]
            # On considère toutes les autres images du même groupe comme les vraies positives attendues
            query_group = set(path_to_group[query_path]) - {query_path}

            # Recherche des k+1 voisins les plus proches de l'image requête
            D_query, I_query = index.search(
                np.array([embeddings[query_idx]]).astype(np.float32), k + 1
            )

            # On filtre les voisins : on garde ceux sous le seuil de distance et différents de la requête
            neighbors = [
                (paths[i], d)
                for i, d in zip(I_query[0], D_query[0])
                if paths[i] != query_path and d < threshold
            ]

            # On récupère uniquement les chemins des voisins sélectionnés
            retrieved_paths = [p for p, _ in neighbors]
            retrieved_set = set(retrieved_paths)

            # Calcul des vrais positifs (retrouvés et pertinents), faux positifs et faux négatifs
            tp = len(retrieved_set & query_group)
            fp = len(retrieved_set - query_group)
            fn = len(query_group - retrieved_set)

            TP += tp
            FP += fp
            FN += fn

    # Calcul des métriques globales
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    threshold = float(threshold)
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    return threshold, precision, recall, f1
