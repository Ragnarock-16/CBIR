import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transforms, bad_list_file):
        self.root_dir = root_dir
        self.transforms = transforms

        bad_paths = set()
        if bad_list_file and os.path.isfile(bad_list_file):
            with open(bad_list_file, 'r') as f:
                bad_paths = set(line.strip() for line in f if line.strip())

        bad_paths = set(os.path.abspath(path) for path in bad_paths)

        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
            and os.path.abspath(os.path.join(root_dir, fname)) not in bad_paths
        ]

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        first_transform = self.transforms()
        second_transform = self.transforms()
        return first_transform(img), second_transform(img)
    
    def __len__(self):
        return len(self.image_paths)

class CBIRImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5573, 0.5598, 0.5478],
                                 std=[0.2112, 0.2071, 0.2058])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), path