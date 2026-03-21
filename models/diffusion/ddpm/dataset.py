from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np

# leak and tight are folders' names
# Return image and int label
class LowTimesDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # ViT 224x224 transform
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 保存所有图片路径 & 标签
        self.samples = []  

        # 显式类到标签的映射：tight -> 0, leak -> 1
        self.class_to_idx = {'tight': 0, 'leak': 1}
        # 反向映射，便于查询
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # 遍历文件夹并构造样本列表（跳过非文件、隐藏文件，并按文件名排序）
        class_names = list(self.class_to_idx.keys())

        for cls_name in class_names:
            cls_folder = os.path.join(image_dir, cls_name)
            if not os.path.isdir(cls_folder):
                # 如果文件夹不存在，跳过（可在外部检查）
                continue
            for filename in sorted(os.listdir(cls_folder)):
                if filename.startswith('.'):
                    continue
                img_path = os.path.join(cls_folder, filename)
                if not os.path.isfile(img_path):
                    continue
                # label 使用显式映射
                label = int(self.class_to_idx[cls_name])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # ResNet transform (512x512)
        img_resnet = self.transform(image) if self.transform else image
        # ViT transform (224x224)
        img_vit = self.vit_transform(image)
        label = torch.tensor(label)
        return img_resnet, img_vit, label

    # 新增方法：返回类到标签的映射，方便确认
    def get_class_mapping(self):
        return dict(self.class_to_idx)

def get_dataloader(train_root, val_root, batch_size=16, num_workers=0, pin_memory=False):
    train_transform = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

    val_transform = transforms.Compose([
    transforms.RandomResizedCrop((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

    train_dataset = LowTimesDataset(image_dir=train_root, transform=train_transform)
    val_dataset = LowTimesDataset(image_dir=val_root, transform=val_transform)
    # print("dataset len=", len(train_dataset))
    # 过采样
    # 统计每个类别样本数
    # targets = [int(label) for _, label in train_dataset.samples]  # [0,0,0,1,1,...]
    # class_sample_count = np.array([len(np.where(np.array(targets)==t)[0]) for t in np.unique(targets)])

    # weights = 1. / class_sample_count
    # samples_weight = np.array([weights[t] for t in targets])

    # # 转成 torch tensor
    # samples_weight = torch.from_numpy(samples_weight).double()
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def get_inference_transforms(res_size=512, vit_size=224):
    """
    返回推理时使用的变换：
      - res_transform: 供 ResNet 分支使用 (默认 512x512，带 normalize)
      - vit_transform: 供 ViT 分支使用 (默认 224x224，带 normalize)
    与训练时的 Normalize 保持一致。
    """
    res_transform = transforms.Compose([
        transforms.Resize((res_size, res_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    vit_transform = transforms.Compose([
        transforms.Resize((vit_size, vit_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return res_transform, vit_transform