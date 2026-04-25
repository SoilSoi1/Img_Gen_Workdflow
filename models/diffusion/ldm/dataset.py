"""
Simple image dataset for unconditional LDM training.
Recursively scans image directories with flexible structure.
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnconditionalImageDataset(Dataset):
    """
    Loads images from a directory for unconditional LDM training.
    
    Supports:
    - Arbitrary folder structure (recursively scans subdirectories)
    - Common image formats: .png, .jpg, .jpeg
    - Flexible image sizes with center cropping and resizing
    
    Args:
        data_dir (str): Root directory containing images
        image_size (int): Target image size (square, will center crop first)
        extensions (tuple): File extensions to load
    """
    
    def __init__(self, data_dir, image_size=512, extensions=('.png', '.jpg', '.jpeg')):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.extensions = tuple(e.lower() for e in extensions)
        
        # Recursively find all image files
        self.image_paths = []
        for root, dirs, files in os.walk(self.data_dir):
            for filename in sorted(files):
                if filename.lower().endswith(self.extensions):
                    self.image_paths.append(os.path.join(root, filename))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")
        
        # Image preprocessing: center crop + resize + normalize to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Lambda(self._center_crop),
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # [-1, 1]
        ])
        
        print(f"[Dataset] Found {len(self.image_paths)} images in {data_dir}")
    
    def _center_crop(self, img):
        """Center crop image to square."""
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        return img.crop((left, top, left + crop_size, top + crop_size))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image
