'''usage
# 基本使用（自动选择设备）
python toTorch/prd_from_image_folders.py \
  --reference_dir /path/to/real_images \
  --eval_dirs /path/to/gen_images1 /path/to/gen_images2 \
  --eval_labels model1 model2 \
  --inception_path /path/to/inception_v3.pth

# 指定GPU
python toTorch/prd_from_image_folders.py \
  --reference_dir ... \
  --eval_dirs ... \
  --eval_labels ... \
  --inception_path ... \
  --device cuda

# 强制使用CPU
python toTorch/prd_from_image_folders.py \
  ... \
  --device cpu
'''
# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
# PyTorch Implementation
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import hashlib

import numpy as np
import torch

# 导入自定义的PyTorch Inception网络模块
import inception_torch
# 导入自定义的PRD（精确率-召回率）计算模块
# 使用与TensorFlow版本相同的prd_score模块（因为PRD计算逻辑与框架无关）
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prd_score as prd

# 创建命令行参数解析器
parser = argparse.ArgumentParser(
    description='Assessing Generative Models via Precision and Recall (PyTorch)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 参考图像所在的目录（真实数据）
parser.add_argument('--reference_dir', type=str, required=True,
                    help='directory containing reference images')
# 待评估的模型生成图像所在的目录或多个目录
parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                    help='directory or directories containing images to be '
                    'evaluated')
# 为每个eval_dirs对应的模型标签
parser.add_argument('--eval_labels', type=str, nargs='+', required=True,
                    help='labels for the eval_dirs (must have same size)')
# 聚类中心数量（用于PRD计算中的K-means聚类）
parser.add_argument('--num_clusters', type=int, default=20,
                    help='number of cluster centers to fit')
# 用于计算PRD曲线的角度数量
parser.add_argument('--num_angles', type=int, default=1001,
                    help='number of angles for which to compute PRD, must be '
                         'in [3, 1e6]')
# 独立运行的次数（用于对PRD数据取平均）
parser.add_argument('--num_runs', type=int, default=10,
                    help='number of independent runs over which to average the '
                         'PRD data')
# 最终图表的保存路径（支持.png或.pdf格式）
parser.add_argument('--plot_path', type=str, default=None,
                    help='path for final plot file (can be .png or .pdf)')
# 缓存目录（存储Inception embeddings）
parser.add_argument('--cache_dir', type=str, default='/tmp/prd_cache/',
                    help='cache directory')
# 预训练的Inception.pth文件路径（PyTorch权重格式）
parser.add_argument('--inception_path', type=str,
                    default='/tmp/prd_cache/inception_v3.pth',
                    help='path to pre-trained Inception V3 .pth file')
# 是否输出日志信息
parser.add_argument('--silent', dest='verbose', action='store_false',
                    help='disable logging output')
# 选择使用的设备（cuda或cpu）
parser.add_argument('--device', type=str, default='auto',
                    help='device to use (cuda/cpu/auto). Default: auto')

# 解析命令行参数
args = parser.parse_args()


def get_device():
    """
    获取计算设备
    """
    if args.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device.lower() in ['cuda', 'cpu']:
        device = torch.device(args.device.lower())
        # 验证设备可用性
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA不可用，回退到CPU")
            return torch.device('cpu')
        return device
    else:
        raise ValueError(f"无效的设备选择: {args.device}")


# 函数：将图像集合通过Inception网络生成特征向量（embeddings）
# 参数：
#   - imgs: 输入的图像数组，形状 [N, H, W, 3]，像素值范围 [0, 255]
#   - inception_path: 预训练Inception模型的路径（.pth格式）
#   - layer_name: 要提取的网络层名称（默认为'avg_pool'，对应TF的'pool_3:0'）
# 返回：Inception特征向量，numpy数组，形状 [N, 2048]
def generate_inception_embedding(imgs, inception_path, layer_name='avg_pool'):
    """
    生成图像的Inception embeddings
    
    对应原始TensorFlow版本，但使用PyTorch实现
    """
    return inception_torch.embed_images_in_inception(
        imgs, inception_path, layer_name, batch_size=32)


# 函数：从缓存加载或生成指定目录中所有图像的Inception embeddings
# 参数：
#   - directory: 图像所在的目录
#   - cache_dir: 缓存目录
#   - inception_path: 预训练Inception模型的路径
# 返回：Inception特征向量
def load_or_generate_inception_embedding(directory, cache_dir, inception_path):
    """
    从缓存加载或生成指定目录中所有图像的Inception embeddings
    
    缓存机制：根据目录路径的MD5哈希值创建缓存文件，避免重复计算
    """
    # 根据目录路径生成MD5哈希值，作为缓存文件的唯一标识符
    hash_value = hashlib.md5(directory.encode('utf-8')).hexdigest()
    # 拼接缓存文件的完整路径
    cache_path = os.path.join(cache_dir, hash_value + '.npy')
    
    # 检查缓存文件是否存在
    if os.path.exists(cache_path):
        # 如果缓存存在，直接加载并返回
        if args.verbose:
            print(f'  → 从缓存加载 embeddings: {cache_path}')
        embeddings = np.load(cache_path)
        if args.verbose:
            print(f'  ✓ 加载成功，shape: {embeddings.shape}')
        return embeddings
    
    # 缓存不存在，需要重新计算
    if args.verbose:
        print('  → 缓存未找到，计算新的embeddings...')
    
    # 调用load_images_from_dir函数加载目录中的所有图像
    imgs = load_images_from_dir(directory)
    if args.verbose:
        print(f'  ✓ 加载了 {len(imgs)} 张图像，形状: {imgs.shape}')
    
    # 调用generate_inception_embedding函数生成embeddings
    embeddings = generate_inception_embedding(imgs, inception_path)
    
    # 创建缓存目录（如果不存在）
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 将计算出的embeddings保存到缓存文件中
    with open(cache_path, 'wb') as f:
        np.save(f, embeddings)
    
    if args.verbose:
        print(f'Embeddings已保存到缓存: {cache_path}')
        print(f'Embeddings形状: {embeddings.shape}')
    
    return embeddings


# 函数：从指定目录加载所有支持格式的图像
# 参数：
#   - directory: 图像所在的目录
#   - types: 支持的图像格式后缀（默认：png, jpg, jpeg, bmp, gif）
# 返回：图像数组（形状为[N, H, W, 3]，像素值范围[0, 255]）
def load_images_from_dir(directory, types=('png', 'jpg', 'jpeg', 'bmp', 'gif')):
    """
    从指定目录加载所有支持格式的图像
    
    使用OpenCV读取图像，并从BGR转换为RGB格式
    """
    # 列出目录中所有支持格式的图像文件路径
    paths = []
    for fn in os.listdir(directory):
        ext = os.path.splitext(fn)[-1][1:].lower()
        if ext in types:
            paths.append(os.path.join(directory, fn))
    
    if not paths:
        raise ValueError(f"目录 '{directory}' 中未找到图像文件")
    
    # 读取并转换图像格式（BGR→RGB）
    # OpenCV默认使用BGR格式，需要转换为RGB格式用于后续处理
    # 注：图像像素值范围为[0, 255]
    imgs = []
    for path in paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️  警告: 无法读取图像 {path}，跳过")
                continue
            # 转换BGR到RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img_rgb)
        except Exception as e:
            print(f"⚠️  警告: 处理图像 {path} 出错: {e}，跳过")
            continue
    
    if not imgs:
        raise ValueError(f"未能成功加载任何图像（已检查 {len(paths)} 个文件）")
    
    # 将列表转换为numpy数组并返回
    return np.array(imgs, dtype=np.float32)


# 主程序入口
if __name__ == '__main__':
    # 检查eval_dirs和eval_labels数量是否匹配
    if len(args.eval_dirs) != len(args.eval_labels):
        raise ValueError(
            'Number of --eval_dirs must be equal to number of --eval_labels.')

    # 将所有目录路径转换为绝对路径
    reference_dir = os.path.abspath(args.reference_dir)
    eval_dirs = [os.path.abspath(directory) for directory in args.eval_dirs]

    # 验证所有目录都存在
    for directory in [reference_dir] + eval_dirs:
        if not os.path.isdir(directory):
            raise ValueError(f"目录不存在: {directory}")

    # 获取计算设备
    device = get_device()
    if args.verbose:
        print(f"\n使用设备: {device}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n" + "="*70)
    print("精确率-召回率分布评估 (Precision-Recall Distribution Evaluation)")
    print("="*70)

    # 第一步：加载或生成参考图像（真实数据）的Inception embeddings
    if args.verbose:
        print(f'\n第一步: 计算参考图像的 Inception embeddings')
        print(f'目录: {reference_dir}')
    # 调用load_or_generate_inception_embedding函数生成/加载参考数据的embeddings
    real_embeddings = load_or_generate_inception_embedding(
        reference_dir, args.cache_dir, args.inception_path)
    
    # 初始化列表，用于存储每个模型的PRD数据
    prd_data = []
    
    # 第二步：逐个处理每个待评估的模型目录
    for idx, directory in enumerate(eval_dirs):
        if args.verbose:
            print(f'\n第二步 {idx+1}/{len(eval_dirs)}: 计算生成图像的 Inception embeddings')
            print(f'目录: {directory}')
        
        # 调用load_or_generate_inception_embedding函数生成/加载待评估数据的embeddings
        eval_embeddings = load_or_generate_inception_embedding(
            directory, args.cache_dir, args.inception_path)
        
        if args.verbose:
            print(f'\n第三步 {idx+1}/{len(eval_dirs)}: 计算 PRD 曲线')
        
        # 第三步：调用prd_score模块中的compute_prd_from_embedding函数计算PRD
        # 该函数使用eval_embeddings（生成的图像特征）和real_embeddings（真实图像特征）
        # 返回精确率和召回率的曲线数据
        prd_data.append(prd.compute_prd_from_embedding(
            eval_data=eval_embeddings,
            ref_data=real_embeddings,
            num_clusters=args.num_clusters,
            num_angles=args.num_angles,
            num_runs=args.num_runs))
    
    if args.verbose:
        print(f'\n第四步: 生成结果图表')

    print()
    
    # 第四步：计算F-beta分数（精确率和召回率的加权调和平均）
    # beta=8表示更重视召回率（召回率权重为精确率的8倍）
    # 调用prd.prd_to_max_f_beta_pair函数将PRD数据转换为F-beta分数
    f_beta_data = [prd.prd_to_max_f_beta_pair(precision, recall, beta=8)
                   for precision, recall in prd_data]
    
    # 打印评估结果表头
    print("="*70)
    print('评估结果 (Evaluation Results)')
    print("="*70)
    print('F_8   F_1/8     模型 (Model)')
    print("-"*70)
    
    # 逐个打印每个模型的F-beta分数
    # F_8：beta=8的F分数（更重视召回率）
    # F_1/8：beta=1/8的F分数（更重视精确率）
    for directory, label, f_beta in zip(eval_dirs, args.eval_labels, f_beta_data):
        print('%.3f  %.3f     %s' % (f_beta[0], f_beta[1], label))
    
    print("="*70)
    print("\n注释:")
    print("  F_8   - F分数 (β=8)，更重视召回率")
    print("  F_1/8 - F分数 (β=1/8)，更重视精确率")

    # 第五步：调用prd.plot函数绘制PRD曲线并保存
    # 根据--plot_path参数决定是否保存图表
    if args.verbose:
        print(f'\n绘制 PRD 曲线...')
    prd.plot(prd_data, labels=args.eval_labels, out_path=args.plot_path)
    
    if args.plot_path:
        print(f'✓ 图表已保存到: {args.plot_path}')
    else:
        print('✓ 图表已显示')
