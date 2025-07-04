from PIL import Image
import matplotlib.pyplot as plt
from dataset.augment import Augmentation
import torch
from torchvision import transforms
import argparse

def test_augmentations(image_path, augmentation_obj):
    img = Image.open(image_path).convert('RGB')

    all_augs = augmentation_obj.augmentations

    n = len(all_augs)
    plt.figure(figsize=(3 * n, 5), facecolor='black')

    aug_titles = [
        'Rand crop + flip', 'Rand crop + Blur', 'Gaussian Noise',
        'Gaussian Blur', 'B/W contrast', 'Padding + Rotation',
        'Book crease', 'Color jitter', 'Sketch effect'
    ]

    for i, aug_func in enumerate(all_augs):
        if callable(aug_func):
            out = aug_func(img)
        else:
            print(f"Augmentation {i+1} is not callable.")
            continue

        # Convert tensor to PIL image for display
        if isinstance(out, torch.Tensor):
            mean = torch.tensor([0.5573, 0.5598, 0.5478]).view(3, 1, 1)
            std = torch.tensor([0.2112, 0.2071, 0.2058]).view(3, 1, 1)
            img_tensor = out * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            out = transforms.ToPILImage()(img_tensor)
            print(f"Augmentation {i+1}: Image size after transform = {out.size}")

        plt.subplot(1, n, i + 1)
        plt.imshow(out)
        plt.axis('off')
        title = aug_titles[i] if i < len(aug_titles) else f'Aug {i+1}'
        plt.title(title, color='white')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image augmentations visually.')
    parser.add_argument('-i', '--image_path', type=str, required=True,
                        help='Path to the image to visualize augmentations.')
    args = parser.parse_args()

    mean = [0.5573, 0.5598, 0.5478]
    std = [0.2112, 0.2071, 0.2058]
    size = 224

    augmentation_obj = Augmentation(mean, std, size)
    test_augmentations(args.image_path, augmentation_obj)
