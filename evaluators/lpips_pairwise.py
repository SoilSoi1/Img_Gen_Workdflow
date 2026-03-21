import os
import torch
import lpips
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def load_images(image_dir: str, device: str = 'cuda') -> list:
    """加载图像目录中的所有图像，返回张量列表"""
    image_paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    images = []
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0).to(device)
        images.append(img)
    
    return images


def cal_lpips_pairwise(image_dir: str, device: str = 'cuda', net: str = 'alex', sample_pairs: int = None) -> float:
    """
    计算图像数据集中图像对的LPIPS，返回平均分数
    
    参数:
        image_dir (str): 图像文件夹路径
        device (str): 计算设备 ('cuda' 或 'cpu')
        net (str): LPIPS网络类型 ('alex', 'vgg', 'squeeze')
        sample_pairs (int): 随机采样的图像对数，None表示逐个图像与所有其他图像比较
    
    返回:
        float: 图像对的平均LPIPS分数
    """
    print(f"开始计算LPIPS (网络: {net}), 设备: {device}...")
    
    images = load_images(image_dir, device)
    n_images = len(images)
    
    if n_images < 2:
        raise ValueError(f"至少需要2张图像，当前只有{n_images}张")
    
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    scores = []
    
    if sample_pairs is None:
        total_pairs = n_images * (n_images - 1) // 2
        pbar = tqdm(total=total_pairs, desc="计算LPIPS")
        
        for i in range(n_images):
            for j in range(i + 1, n_images):
                with torch.no_grad():
                    score = loss_fn(images[i], images[j]).item()
                scores.append(score)
                pbar.update(1)
        pbar.close()
    else:
        indices = np.arange(n_images)
        pair_list = []
        for i in range(n_images):
            for j in range(i + 1, n_images):
                pair_list.append((i, j))
        
        sample_size = min(sample_pairs, len(pair_list))
        sampled_pairs = np.random.choice(len(pair_list), size=sample_size, replace=False)
        
        pbar = tqdm(total=sample_size, desc="计算LPIPS")
        for idx in sampled_pairs:
            i, j = pair_list[idx]
            with torch.no_grad():
                score = loss_fn(images[i], images[j]).item()
            scores.append(score)
            pbar.update(1)
        pbar.close()
    
    mean_score = sum(scores) / len(scores)
    print(f"LPIPS计算完成，共{len(scores)}对图像")
    
    return mean_score


if __name__ == "__main__":
    IMAGE_DIR = "./evaluators/test_data/gen_img"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.isdir(IMAGE_DIR):
        # sample_pairs=None 表示计算所有配对，或指定数字进行随机采样
        mean_lpips = cal_lpips_pairwise(IMAGE_DIR, device=DEVICE, net='alex', sample_pairs=None)
        print(f"\nMean LPIPS Score: {mean_lpips:.4f}")
    else:
        print(f"Error: Image directory not found: {IMAGE_DIR}")

