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