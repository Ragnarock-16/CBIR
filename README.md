# CBIR Project – Near Duplicate Image Retrieval Using Self-Supervised Learning (SimCLR)

## Context

This project aims to develop a visual representation learning model for Content-Based Image Retrieval (CBIR), specifically targeting the retrieval of near-duplicate images. The application domain focuses on a heritage dataset of historical photographs.

## Objective

Implement and adapt a state-of-the-art self-supervised learning method based on SimCLR to learn effective image embeddings for similarity search in the targeted domain.

## Methods

- **Method :** Self-supervised learning approach adapting SimCLR with a fine tuned RESNET backbone to the specific characteristics of historical and/or medical images.

## Pipeline

1. **Embeddings:**  
   Learn image embeddings using SimCLR.

2. **Indexing:**  
   Store the embeddings in FAISS for efficient similarity search.

3. **Near-Duplicate Search:**  
   Perform similarity search to retrieve near-duplicate images.
## Evaluation Metrics

- **Precision@K:**  
  The proportion of the top K retrieved images that are true near-duplicates.

- **Recall@K:**  
  The proportion of all true near-duplicates that are found in the top K retrieved images. It reflects the model’s ability to find as many relevant images as possible within the top K.

- **F1-score@K:**  
  The harmonic mean of Precision@K and Recall@K, providing a balance between precision and recall in the top K results.

- **Mean Average Precision (mAP)@K:**  
  The average precision computed across all queries considering the top K retrieved images, which summarizes both precision and recall across different recall levels.

## Augmentation Strategy

We implemented and compared two augmentation strategies:

1. **SimCLR Augmentation Strategy:**  
   This follows the original SimCLR approach, which includes strong random augmentations such as Gaussian blur, color jitter, random grayscale conversion, random cropping, and horizontal flipping. These augmentations aim to create diverse positive pairs while preserving the semantic content of the images.

2. **Domain-Informed Augmentation Strategy :**  
   This strategy is tailored to the specific nature of our dataset (historical photographs). It includes augmentations chosen to preserve domain-relevant features and avoid distortions that could alter important details.
   

## Results

The retrieval performance and training parameters for different models on the near-duplicate search task are summarized below:
| Model    | Precision@1 | Recall@1 | Precision@5 | Recall@5 | F1@5   | Precision@10 | Recall@10 | F1@10  | mAP@5  | mAP@10 |
|----------|-------------|----------|-------------|----------|--------|---------------|-----------|--------|--------|--------|
| RESNET-50 | 0.4496      | 0.1337   | 0.241       | 0.2885   | 0.2362 | 0.1475        | 0.3402    | 0.1875 | 0.2483 | 0.2652 |
| DINO     | 0.759       | 0.2457   | 0.5         | 0.6007   | 0.488  | 0.3144        | 0.6905    | 0.3903 | 0.5553 | 0.6088 |

| Model  | Batch Size | Learning Rate | Projection Dim | Temperature | Augmentations | Precision@1 | Recall@1 | Precision@5 | Recall@5 | F1@5   | Precision@10 | Recall@10 | F1@10  | mAP@5  | mAP@10 |
|--------|------------|---------------|----------------|-------------|---------------|-------------|----------|-------------|----------|--------|--------------|-----------|--------|--------|--------|
| run1   | 32         | 0.01          | 64             | 0.5         | Custom        | 0.241       | 0.0602   | 0.1324      | 0.1638   | 0.1323 | 0.0975       | 0.2324    | 0.1265 | 0.1184 | 0.1368 |
| run2   | 128        | 0.001         | 64             | 0.5         | Custom        | 0.446       | 0.1285   | 0.2576      | 0.3249   | 0.2592 | 0.1701       | 0.4071    | 0.2209 | 0.2756 | 0.3032 |
| run3   | 128        | 0.001         | 128            | 0.5         | Custom       | 0.446       | 0.1275   | 0.2345      | 0.2953   | 0.2352 | 0.1745       | 0.4146    | 0.2264 | 0.2411 | 0.2791 |
| run4   | 32         | 0.01          | 64             | 0.5         | SimCLR        | 0.0719      | 0.0142   | 0.0446      | 0.0507   | 0.0432 | 0.0324       | 0.0781    | 0.0417 | 0.0297 | 0.0349 |


## Installation

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
