import os
import torch
from cleanfid import fid


def cal_fid(path_real_images: str, path_fake_images: str, device: str = 'cuda'):
    """
    计算两个图像文件夹之间的 FID 分数。

    参数:
        path_real_images (str): 真实图像文件夹的路径。
        path_fake_images (str): 生成图像文件夹的路径。
        device (str): 计算设备 ('cuda' 或 'cpu')。
    
    返回:
        float: 计算出的 FID 分数。
    """
    print(f"开始计算 FID, 设备: {device}...")
    
    # clean-fid 库的 fid.compute_fid 函数
    # 它会自动加载 Inception 模型，提取特征，并计算 FID 公式
    fid_score = fid.compute_fid(
        path_real_images, 
        path_fake_images,  
        mode="clean",               
        device=device,
        num_workers=8 if device == 'cuda' else 0
    )
    
    print("FID 计算完成。")
    return fid_score


if __name__ == "__main__":
    REAL_PATH = "./evaluators/test_data/real_img"
    FAKE_PATH = "./evaluators/test_data/gen_img"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.isdir(REAL_PATH) and os.path.isdir(FAKE_PATH):
        # 计算 FID
        fid_score = cal_fid(REAL_PATH, FAKE_PATH, device=DEVICE)
        print(f"\nFinal FID Score: {fid_score:.4f}")

    else:
        print(f"\nError: Please create and fill the following two paths with images to run the example:\n Real IMG path: {REAL_PATH}\nGen IMG path: {FAKE_PATH}")