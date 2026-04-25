#!/usr/bin/env python3
"""
快速参考：训练和推理 LDM

这个脚本展示了如何快速上手 LDM 的训练和推理。
"""

import os
import subprocess
import sys
from pathlib import Path

# 基础路径
LDM_DIR = Path(__file__).parent
REPO_ROOT = Path(__file__).parent.parent.parent

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")

def train_example():
    """训练示例"""
    print_header("LDM 训练示例")
    
    print("【基本训练】")
    print("python train.py \\")
    print("  --train_root ./datasets/color_20260321/train \\")
    print("  --image_size 256 \\")
    print("  --epochs 100 \\")
    print("  --batch_size 4\n")
    
    print("【从断点续训】")
    print("python train.py \\")
    print("  --train_root ./datasets/color_20260321/train \\")
    print("  --resume ./experiments/ldm/exp_1/checkpoints/epoch_00050.pt\n")
    
    print("【完整自定义】")
    print("python train.py \\")
    print("  --train_root /path/to/train \\")
    print("  --val_root /path/to/val \\")
    print("  --image_size 256 \\")
    print("  --channels 3 \\")
    print("  --epochs 200 \\")
    print("  --batch_size 8 \\")
    print("  --num_workers 8 \\")
    print("  --base_lr 2.0e-06 \\")
    print("  --scale_lr \\")
    print("  --grad_clip 1.0 \\")
    print("  --save_dir ./experiments/ldm/my_exp \\")
    print("  --ckpt_interval 5 \\")
    print("  --device cuda:0\n")

def infer_example():
    """推理示例"""
    print_header("LDM 推理示例")
    
    print("【基本推理】")
    print("python infer.py \\")
    print("  --ckpt ./experiments/ldm/exp_1/checkpoints/epoch_00100.pt \\")
    print("  --num_samples 10\n")
    
    print("【完整自定义】")
    print("python infer.py \\")
    print("  --ckpt ./experiments/ldm/exp_1/checkpoints/epoch_00100.pt \\")
    print("  --num_samples 10 \\")
    print("  --ddim_steps 100 \\")
    print("  --eta 0.0 \\")
    print("  --seed 42 \\")
    print("  --output_dir ./generated_images \\")
    print("  --device cuda:0\n")

def tensorboard_example():
    """TensorBoard 可视化"""
    print_header("TensorBoard 可视化")
    
    print("在训练过程中，可以用 TensorBoard 实时查看训练进度：\n")
    print("tensorboard --logdir ./experiments/ldm/exp_1/logs\n")
    print("然后在浏览器中访问：http://localhost:6006")
    print("可以监控以下指标：")
    print("  - train/loss: 训练损失")
    print("  - val/loss: 验证损失（如果提供了val_root）")
    print("  - train/lr: 学习率\n")

def file_structure():
    """文件结构说明"""
    print_header("项目文件结构")
    
    print("""
models/diffusion/ldm/
├── __init__.py           # 包初始化
├── dataset.py            # 数据加载模块（UnconditionalImageDataset）
├── train.py              # 训练脚本（LDMTrainer）
├── infer.py              # 推理脚本（LDMInference）
├── util.py               # 工具函数与扩散工具库
├── ddim.py               # DDIM 快速采样器
├── configs/              # 模型配置（YAML）
├── README.md             # 详细文档
└── quick_start.py        # 本文件（快速参考）

输出结构：
experiments/ldm/
└── exp_1/
    ├── checkpoints/      # 保存的模型权重（每 N 个 epoch）
    │   ├── epoch_00010.pt
    │   ├── epoch_00020.pt
    │   └── ...
    ├── logs/             # TensorBoard 日志
    │   └── events.*
    └── samples/          # 生成的样本（可选）
    """)

def key_parameters():
    """关键参数说明"""
    print_header("关键参数说明")
    
    print("【数据参数】")
    print("  --train_root:  训练数据目录（必需）")
    print("  --val_root:    验证数据目录（可选）")
    print("  --image_size:  图片大小，默认 256")
    print("  --channels:    图片通道数，默认 3（RGB）\n")
    
    print("【训练参数】")
    print("  --epochs:      训练轮数，默认 100")
    print("  --batch_size:  批次大小，默认 4")
    print("  --base_lr:     基础学习率，默认 2.0e-06")
    print("  --scale_lr:    是否按 batch_size 缩放 LR（推荐）")
    print("  --grad_clip:   梯度裁剪值，默认 1.0\n")
    
    print("【检查点和日志】")
    print("  --save_dir:       保存目录，自动生成时间戳")
    print("  --ckpt_interval:  检查点保存间隔（epoch），默认 10")
    print("  --log_interval:   日志打印间隔（batch），默认 100\n")
    
    print("【推理参数】")
    print("  --ckpt:        模型权重路径（必需）")
    print("  --num_samples: 生成图片数，默认 4")
    print("  --ddim_steps:  DDIM 采样步数，默认 50")
    print("             （越高越慢但质量更好，常用 50-100）")
    print("  --eta:         DDIM 参数，0=确定性，1=随机性，默认 0.0\n")

def memory_usage():
    """显存估计"""
    print_header("显存估计")
    
    print("【batch_size=4 时的显存占用】")
    print(f"  image_size=128:  ~4GB")
    print(f"  image_size=256:  ~10GB")
    print(f"  image_size=512:  ~30GB+\n")
    
    print("【优化建议】")
    print("  - 显存不足：减小 batch_size 或 image_size")
    print("  - 训练太慢：增加 batch_size 或 num_workers")
    print("  - 质量不好：增加 epoch 或减小 base_lr\n")

def tips():
    """实用技巧"""
    print_header("实用技巧")
    
    print("【数据准备】")
    print("  1. 确保所有图片 >= image_size（推荐 >= 256）")
    print("  2. 支持 PNG 和 JPEG，脚本会自动递归扫描子文件夹")
    print("  3. 数据越多越好（推荐 >1000 张）\n")
    
    print("【训练技巧】")
    print("  1. 先用小 image_size（128）快速测试")
    print("  2. 监控 TensorBoard 中的损失曲线选择合适的 epoch 数")
    print("  3. 如果损失停止下降，可以减小 learning rate\n")
    
    print("【推理技巧】")
    print("  1. eta=0.0 时推理速度快但样本多样性少")
    print("  2. eta=1.0 时样本多样性高但推理速度慢")
    print("  3. 增加 ddim_steps 可以提高质量（建议 50-100）\n")

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            train_example()
        elif command == 'infer':
            infer_example()
        elif command == 'tensorboard':
            tensorboard_example()
        elif command == 'files':
            file_structure()
        elif command == 'params':
            key_parameters()
        elif command == 'memory':
            memory_usage()
        elif command == 'tips':
            tips()
        elif command == 'all':
            train_example()
            infer_example()
            tensorboard_example()
            file_structure()
            key_parameters()
            memory_usage()
            tips()
        else:
            print(f"未知命令：{command}")
            print_usage()
    else:
        print_usage()

def print_usage():
    """打印使用说明"""
    print_header("LDM 快速参考使用说明")
    
    print("用法：python quick_start.py [命令]\n")
    print("可用命令：")
    print("  train       - 显示训练示例")
    print("  infer       - 显示推理示例")
    print("  tensorboard - 显示 TensorBoard 使用说明")
    print("  files       - 显示文件结构")
    print("  params      - 显示参数说明")
    print("  memory      - 显示显存估计")
    print("  tips        - 显示实用技巧")
    print("  all         - 显示所有内容\n")
    
    print("示例：")
    print("  python quick_start.py train")
    print("  python quick_start.py all\n")

if __name__ == '__main__':
    main()
