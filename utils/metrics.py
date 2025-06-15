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
    
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    D, _ = index.search(embeddings.astype(np.float32), k + 1)

    if threshold_type == "mean":
        threshold = np.mean(D[:, 1:])
    elif threshold_type == "median":
        threshold = np.median(D[:, 1:])
    elif threshold_type == "percentile_25":
        threshold = np.percentile(D[:, 1:], 25)
    else:
        raise ValueError("Unsupported threshold_type")

    TP = 0
    FP = 0
    FN = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue

            query_idx = path_to_index[query_path]
            query_group = set(path_to_group[query_path]) - {query_path}

            D_query, I_query = index.search(
                np.array([embeddings[query_idx]]).astype(np.float32), k + 1
            )

            neighbors = [
                (paths[i], d)
                for i, d in zip(I_query[0], D_query[0])
                if paths[i] != query_path and d > threshold

            ]

            retrieved_paths = [p for p, _ in neighbors]
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

    threshold = float(threshold)
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    return threshold, precision, recall, f1

### METRICS NO FAISS ###

def compute_top_k_neighbors(embeddings: np.ndarray, k: int, query_idx: int):
    query_embedding = embeddings[query_idx]
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    distances[query_idx] = np.inf
    return np.argsort(distances)[:k]


def recall_at_k_no_faiss(embeddings, paths, groups, k=5):
    """
    Calcule le Recall@K sur des groupes d'images similaires (near duplicates).

    Le Recall@K mesure la proportion d'éléments pertinents (du même groupe que l'image requête)
    retrouvés parmi les K plus proches voisins.

    Formule :
        Recall@K = (Nombre d'éléments pertinents dans les K plus proches voisins) / (Nombre total d'éléments pertinents)

    Args:
        embeddings (np.ndarray): Tableau des vecteurs d'embedding (taille N x D).
        paths (List[str]): Liste des chemins d'images correspondant aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins à considérer (top-k).

    Returns:
        float: Moyenne du Recall@K sur toutes les requêtes valides.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_recall = 0.0
    count = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            neighbor_indices = compute_top_k_neighbors(embeddings, k + 1, query_idx)
            retrieved_paths = [paths[i] for i in neighbor_indices if paths[i] != query_path][:k]

            retrieved_relevant = sum(1 for p in retrieved_paths if p in relevant_items)
            total_possible = len(relevant_items)

            if total_possible > 0:
                total_recall += retrieved_relevant / total_possible
                count += 1

    return total_recall / count if count > 0 else 0.0


def precision_at_k_no_faiss(embeddings, paths, groups, k=5):
    """
    Calcule la Precision@K sur les groupes d'images similaires.

    La Precision@K mesure la proportion des K éléments retournés qui sont réellement
    pertinents (du même groupe que l'image requête).

    Formule :
        Precision@K = (Nombre d'éléments pertinents dans les K plus proches voisins) / K

    Args:
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins d'images associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne de la précision à K sur toutes les requêtes valides.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}

    total_precision = 0.0
    count = 0

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            neighbor_indices = compute_top_k_neighbors(embeddings, k + 1, query_idx)
            retrieved_paths = [paths[i] for i in neighbor_indices if paths[i] != query_path][:k]

            retrieved_relevant = sum(1 for p in retrieved_paths if p in relevant_items)
            total_precision += retrieved_relevant / k
            count += 1

    return total_precision / count if count > 0 else 0.0


def mean_average_precision_no_faiss(embeddings, paths, groups, k=10):
    """
    Calcule le Mean Average Precision (mAP) pour des groupes d'images similaires.

    Le mAP tient compte de la position des bons résultats parmi les K premiers.

    Args:
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires.
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne des précisions moyennes (mAP) sur toutes les requêtes.
    """
    path_to_index = {path: i for i, path in enumerate(paths)}
    path_to_group = {path: group for group in groups for path in group}
    average_precisions = []

    for group in groups:
        for query_path in group:
            if query_path not in path_to_index:
                continue
            query_idx = path_to_index[query_path]
            relevant_items = set(path_to_group[query_path]) - {query_path}

            neighbor_indices = compute_top_k_neighbors(embeddings, k + 1, query_idx)
            retrieved_paths = [paths[i] for i in neighbor_indices if paths[i] != query_path][:k]

            relevant_hits = 0
            precision_sum = 0.0

            for rank, retrieved_path in enumerate(retrieved_paths, start=1):
                if retrieved_path in relevant_items:
                    relevant_hits += 1
                    precision_sum += relevant_hits / rank

            if relevant_hits > 0:
                average_precisions.append(precision_sum / len(relevant_items))
            else:
                average_precisions.append(0.0)

    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0


def f1_score_at_k_no_faiss(embeddings, paths, groups, k=5):
    """
    Calcule le F1-Score@K, moyenne harmonique entre la Precision@K et le Recall@K.

    Formule :
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)

    Args:
        embeddings (np.ndarray): Vecteurs d'embedding.
        paths (List[str]): Chemins d'images associés aux embeddings.
        groups (List[List[str]]): Groupes d'images similaires (ground truth).
        k (int): Nombre de voisins considérés.

    Returns:
        float: Moyenne du F1@K sur toutes les requêtes valides.
    """
    precision = precision_at_k_no_faiss(embeddings, paths, groups, k)
    recall = recall_at_k_no_faiss(embeddings, paths, groups, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
