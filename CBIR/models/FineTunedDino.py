import torch
import torch.nn as nn

class FineTunedDino(nn.Module):
    def __init__(self, projection_dim = 64):
        super(FineTunedDino,self).__init__()
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        
        self.projection_head = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Linear(384,projection_dim)
        )
        
    def forward(self, x):
        h = self.encoder.forward_features(x)['x_norm_clstoken']
        z = self.projection_head(h)
        return h,z