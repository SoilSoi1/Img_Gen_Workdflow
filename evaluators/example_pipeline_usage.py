"""
评估管道使用示例
"""

from evaluators.pipeline import EvaluationPipeline

# 示例1: 基础用法 - 计算所有可用的评估指标
pipeline = EvaluationPipeline(device='cpu')
results = pipeline.run(
    gen_dir='./evaluators/test_data/gen_img',
    real_dir='./evaluators/test_data/real_img'
)
print(results)
# 输出: {'fid': 395.1278, 'kid': 0.3570, 'lpips': 0.3664, 'brisque': 31.7136}


# 示例2: 仅计算单数据集方法
pipeline = EvaluationPipeline(device='cpu')
results = pipeline.run(
    gen_dir='./evaluators/test_data/gen_img',
    methods=['lpips', 'brisque'],
    lpips_net='alex'
)
print(results)
# 输出: {'lpips': 0.3664, 'brisque': 31.7136}


# 示例3: 仅计算分布级别的对比方法
pipeline = EvaluationPipeline(device='cpu')
results = pipeline.run(
    gen_dir='./evaluators/test_data/gen_img',
    real_dir='./evaluators/test_data/real_img',
    methods=['fid', 'kid']
)
print(results)
# 输出: {'fid': 395.1278, 'kid': 0.3570}


# 示例4: 自定义LPIPS参数（采样对数）
pipeline = EvaluationPipeline(device='cuda')
results = pipeline.run(
    gen_dir='./evaluators/test_data/gen_img',
    methods=['lpips'],
    lpips_net='vgg',
    lpips_sample_pairs=100  # 仅采样100对，加快计算
)
print(results)


# 命令行用法示例
# ==============

# 1. 计算所有指标
# python evaluators/pipeline.py \
#   --gen_dir ./path/to/generated \
#   --real_dir ./path/to/real \
#   --methods fid,kid,lpips,brisque

# 2. 仅计算单数据集方法
# python evaluators/pipeline.py \
#   --gen_dir ./path/to/generated \
#   --methods lpips,brisque \
#   --lpips_net vgg

# 3. 指定LPIPS采样对数
# python evaluators/pipeline.py \
#   --gen_dir ./path/to/generated \
#   --methods lpips \
#   --lpips_sample_pairs 1000

# 4. 使用GPU计算，输出JSON格式结果
# python evaluators/pipeline.py \
#   --gen_dir ./path/to/generated \
#   --real_dir ./path/to/real \
#   --device cuda \
#   --output json

# 5. 计算PRD（需要Inception权重）
# python evaluators/pipeline.py \
#   --gen_dir ./path/to/generated \
#   --real_dir ./path/to/real \
#   --methods prd \
#   --prd_inception_path ./evaluators/prd/inception_v3.pth


# 方法说明
# ========

# 单数据集方法（只需gen_dir）:
#   - lpips: 图像对相似度评估，值越低越好，支持采样和网络选择
#   - brisque: 无参考图像质量评估，值越小越好（0-100范围）

# 双数据集方法（需要gen_dir和real_dir）:
#   - fid: Fréchet Inception Distance，值越小越好
#   - kid: Kernel Inception Distance（无偏估计），值越小越好
#   - prd: Precision & Recall for Distribution，返回F-beta分数

# LPIPS网络选择:
#   - alex: AlexNet，最快速
#   - vgg: VGGNet，更精细
#   - squeeze: SqueezeNet，轻量级
