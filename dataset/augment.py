from PIL import ImageEnhance, Image, ImageFilter, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)
    
class EnhanceBlackWhiteContrast:
    def __init__(self, factor=1.5):
        """
        Args:
            factor (float): Contrast factor. 1.0 = original image.
                            >1.0 = more contrast, <1.0 = less.
        """
        self.factor = factor

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Contrast-enhanced image.
        """
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.factor)
    
class RandomPadding:
    def __init__(self, max_padding=20, fill=255):
        self.max_padding = max_padding
        self.fill = fill

    def __call__(self, img):
        pad = random.randint(0, self.max_padding)
        return TF.pad(img, padding=pad, fill=self.fill)
    

class GradientBookCrease:
    def __init__(self, strength=0.3, width_ratio=0.2):
        """
        Args:
            strength (float): Maximum darkness at the center (0 to 1).
            width_ratio (float): Width of the crease relative to image width (e.g. 0.2 = 20%).
        """
        self.strength = strength
        self.width_ratio = width_ratio

    def __call__(self, img):
        img = img.convert("RGB")
        w, h = img.size
        center = w // 2
        half_width = int(self.width_ratio * w // 2)

        dist = np.abs(np.arange(w) - center)
        gradient_1d = np.clip(dist / half_width, 0, 1)

        mask = np.tile(gradient_1d, (h, 1))

        shadow = (mask * 255).astype(np.uint8)
        shadow_rgb = np.stack([shadow] * 3, axis=2)

        shadow_img = Image.fromarray(shadow_rgb)

        return Image.blend(img, shadow_img, alpha=self.strength)

class SketchEffect:
    def __call__(self, img):
        img = img.convert("L")
        edges = img.filter(ImageFilter.FIND_EDGES)
        inverted = ImageOps.invert(edges)
        contrasted = ImageEnhance.Contrast(inverted).enhance(10)
        return contrasted.convert("RGB")

class Augmentation:
      def __init__(self, mean, std, size):
        self.mean = mean
        self.std = std
        self.size = size

        self.augmentations = [
             transforms.Compose([
                transforms.RandomResizedCrop(size,scale=(0.6,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.RandomResizedCrop(size,scale=(0.6,1.0)),
                transforms.GaussianBlur(5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                AddGaussianNoise(std=0.07),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.Resize((size,size)),
                transforms.GaussianBlur(5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.Resize((size,size)),
                EnhanceBlackWhiteContrast(factor=1.8),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                RandomPadding(max_padding=20, fill=0),
                transforms.Resize((size,size)),
                transforms.RandomRotation(10, fill=0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                GradientBookCrease(strength=0.3, width_ratio=0.2),
                transforms.Resize((size,size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.Resize((size,size)),
                transforms.ColorJitter(brightness=.5, hue=.3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.Resize((size,size)),
                SketchEffect(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),


        ]

      def __call__(self):
           return random.choice(self.augmentations)