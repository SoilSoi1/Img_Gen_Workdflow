import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, '/opt/miniconda3/envs/rs1/lib/python3.10/site-packages')
from brisque.brisque import BRISQUE


def cal_brisque_official(image_dir: str) -> float:
    """
    使用官方BRISQUE库计算图像数据集的BRISQUE分数，返回平均值
    
    参数:
        image_dir (str): 图像文件夹路径
    
    返回:
        float: 数据集中所有图像的平均BRISQUE分数
    """
    print(f"开始计算BRISQUE（官方库）...")
    
    image_paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
    
    if len(image_paths) == 0:
        raise ValueError(f"未在 {image_dir} 中找到图像文件")
    
    model = BRISQUE()
    scores = []
    
    pbar = tqdm(total=len(image_paths), desc="计算BRISQUE")
    
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            img_array = np.array(img)
            score = model.score(img_array)
            scores.append(score)
            pbar.update(1)
        except Exception as e:
            print(f"  处理 {path.name} 失败: {e}")
            pbar.update(1)
    
    pbar.close()
    
    if len(scores) == 0:
        raise ValueError("未能成功处理任何图像")
    
    mean_score = sum(scores) / len(scores)
    print(f"BRISQUE计算完成，共{len(scores)}张图像")
    
    return mean_score


if __name__ == "__main__":
    IMAGE_DIR = "./evaluators/test_data/gen_img"
    
    if os.path.isdir(IMAGE_DIR):
        mean_brisque = cal_brisque_official(IMAGE_DIR)
        print(f"\nMean BRISQUE Score: {mean_brisque:.4f}")
    else:
        print(f"Error: Image directory not found: {IMAGE_DIR}")
