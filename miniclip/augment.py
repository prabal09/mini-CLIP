from torchvision import transforms

# ImageNet channel statistics — fine default for general natural photos.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_train_transform(image_size: int):
    """Training augmentations.

    RandomResizedCrop: takes a random crop of varying scale and resizes
    to image_size. Workhorse augmentation for ViT — forces invariance to
    framing and scale, which directly attacks the spatial shortcut
    learning we observed in step 3.

    Horizontal flip: free 2× data on natural photos.

    ColorJitter: small brightness/contrast/saturation perturbations break
    shortcuts that rely on exact color statistics. Especially relevant
    since our step-3 model used color as the dominant signal.

    Normalize: zero-mean unit-variance per channel using ImageNet stats.
    Standard regardless of the dataset; the network just needs *some*
    consistent normalization.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_eval_transform(image_size: int):
    """Deterministic eval transform: resize then center-crop."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
