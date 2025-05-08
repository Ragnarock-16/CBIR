import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim = 64):
        super(SimCLR,self).__init__()
        
        self.encoder = base_encoder
        dim_mlp = self.encoder.fc.in_features

        self.encoder.fc  = nn.Identity() 

        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp,projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x) # note: Use it for CBIR afterward
        z = self.projection_head(h)
        return h,z
    
class NTXentLoss(nn.Module):
    def __init__(self, device, temperature = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
   
    def forward(self, z_i,z_j):
        batch_size = z_i.shape[0]
        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)


        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
               
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
