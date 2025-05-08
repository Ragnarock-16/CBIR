import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import UnidentifiedImageError

class SimCLRDataset(Dataset):
    def __init__(self, root_dir, base_transform, transform, bad_list_file="/Users/nour/Desktop/MSV/corrupted_images.txt"):
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.transform = transform

        # Load bad image paths from file
        bad_paths = set()
        if bad_list_file and os.path.isfile(bad_list_file):
            with open(bad_list_file, 'r') as f:
                bad_paths = set(line.strip() for line in f if line.strip())

        # Normalize bad paths: absolute, and lowercase if needed
        bad_paths = set(os.path.abspath(path) for path in bad_paths)

        # Collect image paths, skipping known bad ones
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            and os.path.abspath(os.path.join(root_dir, fname)) not in bad_paths
        ]

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.base_transform(img), self.transform(img)
        except:
            print(f"Warning: Skipping corrupted image: {path}")
            return self.__getitem__((idx + 1) % len(self))
        
    def __len__(self):
        return len(self.image_paths)
