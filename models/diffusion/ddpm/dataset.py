from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

# Simple dataset for training
# Returns only the image tensor (no labels, no ViT branch)
class LowTimesDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # 保存所有图片路径
        self.samples = []  

        # 递归遍历所有子文件夹中的图片（不限制于tight/leak）
        for root, dirs, files in os.walk(image_dir):
            for filename in sorted(files):
                if filename.startswith('.'):
                    continue
                img_path = os.path.join(root, filename)
                if os.path.isfile(img_path):
                    self.samples.append(img_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # 使用 train_transform 处理图像
        img_resnet = self.transform(image) if self.transform else image
        # 返回三个值以兼容现有的 train 脚本
        return img_resnet

def get_dataloader(train_root, val_root=None, batch_size=16, num_workers=0, pin_memory=False):
    train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

    train_dataset = LowTimesDataset(image_dir=train_root, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    
    # 返回 None 作为 val_loader，兼容现有脚本
    val_loader = None

    return train_loader, val_loader