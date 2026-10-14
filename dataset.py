import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, max_std=0.08, p=0.4):
        self.mean = mean
        self.max_std = max_std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            std = random.uniform(0.01, self.max_std)
            noise = torch.randn_like(tensor) * std + self.mean
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor

def get_transforms(split="train"):
    if split == "train":
        return T.Compose([
            T.Resize((256, 256)),
            # 1. Perspective tilt & warp (photograph from angled viewpoint)
            T.RandomPerspective(distortion_scale=0.35, p=0.6),
            # 2. Geometric affine (rotation, scale, shear)
            T.RandomAffine(
                degrees=(-20, 20),
                translate=(0.12, 0.12),
                scale=(0.75, 1.25),
                shear=(-12, 12)
            ),
            # 3. Aggressive crop (zooming in on UI elements or full view)
            T.RandomResizedCrop(224, scale=(0.35, 1.0), ratio=(0.7, 1.4)),
            T.RandomHorizontalFlip(p=0.5),
            # 4. Color, lighting, monitor gamuts
            T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45, hue=0.15),
            T.RandomAutocontrast(p=0.3),
            T.RandomPosterize(bits=4, p=0.2),
            T.RandomSolarize(threshold=190, p=0.15),
            # 5. Blur / camera defocus
            T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.5)),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
            # 6. Tensor conversion and sensor noise injection
            T.ToTensor(),
            AddGaussianNoise(p=0.4),
            # 7. Normalization
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # 8. Multi-cutout random erasing (forces model to learn without relying on any single logo or word)
            T.RandomErasing(p=0.6, scale=(0.02, 0.35), ratio=(0.3, 3.3), value="random"),
            T.RandomErasing(p=0.4, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0)
        ])
    else:
        return T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

class MrBeastScamDataset(Dataset):
    def __init__(self, manifest_csv="dataset_manifest.csv", split="train", transform=None):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or get_transforms(split)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = float(row["label"])
        
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224), color=(0, 0, 0))
            
        tensor_img = self.transform(im)
        return tensor_img, torch.tensor(label, dtype=torch.float32)
