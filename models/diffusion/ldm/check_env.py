#!/usr/bin/env python3
"""
检查 LDM 环境和依赖是否正确安装。
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """检查 Python 版本"""
    print("[Check] Python 版本...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ 需要 Python 3.8+，当前 {version.major}.{version.minor}.{version.micro}")
        return False

def check_pytorch():
    """检查 PyTorch"""
    print("[Check] PyTorch...", end=" ")
    try:
        import torch
        import torchvision
        print(f"✓ torch={torch.__version__}, torchvision={torchvision.__version__}")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"[Check] GPU 支持...", end=" ")
            print(f"✓ {torch.cuda.get_device_name(0)}")
            print(f"[Check] 显存...", end=" ")
            print(f"✓ {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("[Warn] 未检测到 CUDA GPU")
        
        return True
    except ImportError as e:
        print(f"✗ {e}")
        return False

def check_ldm_repo():
    """检查 LDM 本地模块可用性"""
    print("[Check] LDM 本地模块...", end=" ")
    
    ldm_path = Path(__file__).parent
    
    try:
        from util import instantiate_from_config
        from ddim import DDIMSampler
        print(f"✓ 位于 {ldm_path}")
        return True
    except ImportError as e:
        print(f"✗ 无法导入: {e}")
        return False

def check_required_packages():
    """检查必需的包"""
    packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow',
        'omegaconf': 'omegaconf',
        'pytorch_lightning': 'pytorch-lightning',
        'tensorboard': 'tensorboard',
    }
    
    all_ok = True
    for module_name, package_name in packages.items():
        print(f"[Check] {package_name}...", end=" ")
        try:
            __import__(module_name)
            print("✓")
        except ImportError:
            print(f"✗ (run: pip install {package_name})")
            all_ok = False
    
    return all_ok

def check_ldm_modules():
    """检查 LDM 内部模块"""
    print("[Check] LDM 内部模块...")
    
    modules = [
        'dataset',
        'util',
        'ddim',
        'train',
        'infer',
    ]
    
    all_ok = True
    for module in modules:
        print(f"  {module}...", end=" ")
        try:
            __import__(module)
            print("✓")
        except (ImportError, ModuleNotFoundError) as e:
            print(f"✗ ({e})")
            all_ok = False
    
    return all_ok

def check_data_directory():
    """检查数据目录"""
    print("[Check] 数据目录...", end=" ")
    
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "color_20260321" / "train"
    
    if data_dir.exists():
        # Count images
        import glob
        images = list(data_dir.glob("**/*.png")) + list(data_dir.glob("**/*.jpg"))
        print(f"✓ 找到 {len(images)} 张图片")
        return True
    else:
        print(f"✗ 数据目录不存在")
        print(f"  预期路径: {data_dir}")
        return False

def check_local_modules():
    """检查本地模块"""
    print("[Check] 本地模块...")
    
    local_dir = Path(__file__).parent
    modules = [
        'dataset',
        'train',
        'infer',
    ]
    
    all_ok = True
    for module in modules:
        module_path = local_dir / f"{module}.py"
        print(f"  {module}.py...", end=" ")
        if module_path.exists():
            print("✓")
        else:
            print(f"✗ 不存在")
            all_ok = False
    
    return all_ok

def main():
    """运行所有检查"""
    print("\n" + "="*60)
    print("LDM 环境检查")
    print("="*60 + "\n")
    
    checks = [
        ("Python 版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("必需的包", check_required_packages),
        ("LDM 官方仓库", check_ldm_repo),
        ("LDM 内部模块", check_ldm_modules),
        ("本地模块", check_local_modules),
        ("数据目录", check_data_directory),
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n[{check_name}]")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ✗ 检查失败: {e}")
            results.append((check_name, False))
    
    # 打印总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    all_ok = all(result for _, result in results)
    
    if all_ok:
        print("\n✓ 所有检查通过，可以开始训练！")
        print("\n快速开始：")
        print("  python train.py --train_root ./datasets/color_20260321/train")
        return 0
    else:
        print("\n✗ 存在未通过的检查，请修复后重试")
        return 1

if __name__ == '__main__':
    sys.exit(main())
