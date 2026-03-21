"""
快速训练脚本 - DDPM 扩散模型

特性:
  1. 通过迭代次数而非 epoch 定义训练量
  2. 自动计算所需 epoch 数 (迭代次数 / 每 epoch 迭代数)
  3. 灵活配置 checkpoints 文件夹命名
  4. 创建带时间戳的训练日志
  5. 支持断点续训

使用示例:
  python train_quick.py --total_iterations 50000 --ckpt_name "exp_1" --ckpt_interval 5000
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# 计算数据集大小的辅助函数
def count_images_in_dir(dir_path):
    """统计文件夹中的图像数量"""
    if not os.path.isdir(dir_path):
        return 0
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    count = sum(1 for f in os.listdir(dir_path) if os.path.splitext(f)[1] in valid_extensions)
    return count


def setup_training_config(args):
    """
    基于命令行参数和数据集大小设置训练配置
    支持两种训练模式: epoch 模式 或 iteration 模式
    
    返回:
        dict: 更新后的 modelConfig
    """
    from Main import modelConfig
    
    # ========== 数据集路径配置 ==========
    modelConfig["train_root"] = args.train_root
    modelConfig["val_root"] = args.val_root
    print(f"✓ 训练数据: {args.train_root}")
    print(f"✓ 验证数据: {args.val_root}")
    
    # ========== 计算迭代相关参数 ==========
    # 1. 计算每 epoch 的迭代次数
    train_img_count = count_images_in_dir(args.train_root)
    batch_size = args.batch_size
    
    if train_img_count == 0:
        print(f"❌ 错误: 在 {args.train_root} 中未找到训练图像!")
        sys.exit(1)
    
    iterations_per_epoch = (train_img_count + batch_size - 1) // batch_size  # ceiling division
    print(f"👉 每个 epoch 的迭代次数: {iterations_per_epoch}")
    print(f"   (数据集: {train_img_count} 张图 / 批大小: {batch_size})")
    
    # 2. 根据 epoch 或 iteration 模式计算所需 epoch 数
    if args.use_epoch is not None:
        # === EPOCH 模式 ===
        required_epochs = args.use_epoch
        actual_iterations = required_epochs * iterations_per_epoch
        
        print(f"\n📊 训练计划 (EPOCH 模式):")
        print(f"   指定 epoch 数:   {required_epochs}")
        print(f"   总迭代次数:      {actual_iterations:,}")
        
    else:
        # === ITERATION 模式 (默认) ===
        total_iterations = args.total_iterations if args.total_iterations is not None else 50000
        required_epochs = (total_iterations + iterations_per_epoch - 1) // iterations_per_epoch
        actual_iterations = required_epochs * iterations_per_epoch
        
        print(f"\n📊 训练计划 (ITERATION 模式):")
        print(f"   目标总迭代次数: {total_iterations:,}")
        print(f"   所需 epoch 数:  {required_epochs}")
        print(f"   实际总迭代次数: {actual_iterations:,} (比目标多 {actual_iterations - total_iterations:,})")
    
    # 3. 更新 modelConfig
    modelConfig["epoch"] = required_epochs
    modelConfig["batch_size"] = batch_size
    
    # ========== Checkpoints 文件夹配置 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.ckpt_name:
        # 自定义名称
        ckpt_dir = os.path.join(args.ckpt_root, args.ckpt_name, timestamp)
    else:
        # 默认使用时间戳
        ckpt_dir = os.path.join(args.ckpt_root, timestamp)
    
    os.makedirs(ckpt_dir, exist_ok=True)
    modelConfig["save_weight_dir"] = ckpt_dir
    print(f"\n💾 Checkpoints 保存位置: {ckpt_dir}")
    
    # ========== 设置 checkpoints 保存间隔 ==========
    if args.ckpt_interval > 0:
        # 根据迭代次数计算保存间隔 (epoch)
        ckpt_interval_epochs = (args.ckpt_interval + iterations_per_epoch - 1) // iterations_per_epoch
        ckpt_interval_epochs = max(1, ckpt_interval_epochs)  # 至少每 1 epoch 保存一次
        
        print(f"💾 检查点保存间隔:")
        print(f"   每 {args.ckpt_interval:,} 迭代保存一次")
        print(f"   ≈ 每 {ckpt_interval_epochs} epoch 保存一次")
        
        # 存储到 config 中供训练循环使用
        modelConfig["ckpt_interval_epochs"] = ckpt_interval_epochs
        modelConfig["ckpt_interval_iterations"] = args.ckpt_interval
    else:
        print(f"💾 检查点保存间隔: 仅保存最后的权重")
        modelConfig["ckpt_interval_epochs"] = None
        modelConfig["ckpt_interval_iterations"] = 0
    
    # ========== 训练参数 ==========
    print(f"\n⚙️  训练参数:")
    print(f"   学习率: {modelConfig['lr']}")
    print(f"   总 batch: {modelConfig['batch_size']}")
    print(f"   T (扩散步数): {modelConfig['T']}")
    print(f"   设备: {modelConfig['device']}")
    
    return modelConfig


def main():
    parser = argparse.ArgumentParser(
        description="快速训练 DDPM 扩散模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # ===== ITERATION 模式 (基于总迭代次数) =====
  
  # 基础使用 - 训练共 50,000 次迭代
  python train_quick.py --total_iterations 50000

  # 自定义 checkpoints 名称
  python train_quick.py --total_iterations 50000 --ckpt_name "exp_1"

  # 每 5,000 次迭代保存一次
  python train_quick.py --total_iterations 50000 --ckpt_name "exp_1" --ckpt_interval 5000

  # 自定义数据路径
  python train_quick.py --total_iterations 50000 \\
    --train_root /path/to/train \\
    --val_root /path/to/val \\
    --ckpt_name "custom_exp"

  # ===== EPOCH 模式 (基于 epoch 数) =====
  
  # 直接指定 epoch 数 (不需要计算)
  python train_quick.py --epoch 100

  # 使用 epoch 模式 + 自定义 checkpoint 名称
  python train_quick.py --epoch 100 --ckpt_name "exp_1"

  # 使用 epoch 模式 + 自定义数据路径
  python train_quick.py --epoch 100 \\
    --train_root /path/to/train \\
    --val_root /path/to/val \\
    --ckpt_name "my_exp"

  # 注意: --total_iterations 和 --epoch 互斥，只能选择其中一个
        """
    )
    
    # ========== 迭代次数相关参数 (互斥: 选择 epoch 或 iteration 模式) ==========
    train_mode_group = parser.add_mutually_exclusive_group()
    
    train_mode_group.add_argument(
        "--total_iterations",
        type=int,
        default=None,
        help="总迭代次数 (与 --epoch 二选一，默认模式)"
    )
    
    train_mode_group.add_argument(
        "--epoch",
        type=int,
        dest="use_epoch",
        help="训练总 epoch 数 (与 --total_iterations 二选一)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="批大小 (默认: 2)"
    )
    
    # ========== 数据集路径 ==========
    parser.add_argument(
        "--train_root",
        type=str,
        default="./newest_data/train/",
        help="训练数据根目录 (默认: ./newest_data/train/)"
    )
    
    parser.add_argument(
        "--val_root",
        type=str,
        default="./newest_data/val/",
        help="验证数据根目录 (默认: ./newest_data/val/)"
    )
    
    # ========== Checkpoints 配置 ==========
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="./Checkpoints/",
        help="Checkpoints 根目录 (默认: ./Checkpoints/)"
    )
    
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default=None,
        help="自定义 checkpoints 文件夹名称 (可选，默认仅用时间戳)"
    )
    
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10000,
        help="检查点保存间隔(迭代次数，0=仅保存最后的权重，默认: 10000)"
    )
    
    # ========== 其他训练参数 ==========
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="断点续训 - 指定要加载的检查点文件"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅进行配置检查，不开始训练"
    )
    
    args = parser.parse_args()
    
    # ========== 配置打印头 ==========
    print("\n" + "="*70)
    print("🚀 DDPM 快速训练脚本")
    print("="*70)
    
    # ========== 设置配置 ==========
    modelConfig = setup_training_config(args)
    
    # ========== 断点续训设置 ==========
    if args.resume:
        modelConfig["training_load_weight"] = args.resume
        print(f"\n📂 断点续训: 加载 {args.resume}")
    
    # ========== 配置摘要 ==========
    print(f"\n📋 配置摘要:")
    print(f"   状态: train")
    print(f"   总 epoch 数: {modelConfig['epoch']}")
    print(f"   日志将保存到: {modelConfig['save_weight_dir']}/training_log_*.json")
    print("="*70 + "\n")
    
    if args.dry_run:
        print("✓ 配置检查完成，--dry_run 模式，无需训练")
        return
    
    # ========== 开始训练 ==========
    print("⏱️  开始训练...\n")
    
    try:
        from Train import train
        
        modelConfig["state"] = "train"
        train(modelConfig)
        
        print("\n" + "="*70)
        print("✅ 训练完成!")
        print(f"   Checkpoints: {modelConfig['save_weight_dir']}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
