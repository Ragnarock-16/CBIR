from torchvision import transforms

def simclr_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""

        kernel_size = int(0.1 * size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 3)

        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size),
                                              transforms.ToTensor()])
        return data_transforms

def basic_transform(size):
    basic_transform = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor()])
    return basic_transform