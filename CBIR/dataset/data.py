import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import UnidentifiedImageError

class SimCLRDataset(Dataset):
    def __init__(self, root_dir, base_transform,transform):
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.base_transform(img), self.transform(img)
        except (UnidentifiedImageError, OSError):
            print(f"Warning: Skipping corrupted image: {path}")
            return self.__getitem__((idx + 1) % len(self))
        
    def __len__(self):
        return len(self.image_paths)
