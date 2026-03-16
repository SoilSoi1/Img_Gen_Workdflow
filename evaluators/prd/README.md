# PRD 评估工具 - 使用指南

## 📋 快速开始

### 基本用法

```bash
python prd_from_image_folders.py \
  --reference_dir /path/to/real_images \
  --eval_dirs /path/to/gen_images \
  --eval_labels model_name
```

### 完整示例

```bash
# 基本运行
python prd_from_image_folders.py \
  --reference_dir ./test_data/real_img/ \
  --eval_dirs ./test_data/gen_img/ \
  --eval_labels model_1

# 指定 GPU
python prd_from_image_folders.py \
  --reference_dir ... \
  --eval_dirs ... \
  --eval_labels ... \
  --device cuda

# 多个模型对比
python prd_from_image_folders.py \
  --reference_dir ./test_data/real_img/ \
  --eval_dirs ./model1_output ./model2_output ./model3_output \
  --eval_labels model_1 model_2 model_3
```

## 🔧 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--reference_dir` | 真实图像目录 | ✅ |
| `--eval_dirs` | 生成图像目录（可多个） | ✅ |
| `--eval_labels` | 模型标签（数量需匹配 eval_dirs） | ✅ |
| `--num_clusters` | 聚类中心数（默认20） | ❌ |
| `--num_angles` | PRD 曲线采样点数（默认1001） | ❌ |
| `--num_runs` | 独立运行次数（默认10） | ❌ |
| `--plot_path` | 结果图表保存路径 | ❌ |
| `--cache_dir` | 缓存目录（默认当前目录） | ❌ |
| `--inception_path` | Inception V3 权重路径 | ❌ |
| `--device` | 计算设备（cuda/cpu/auto） | ❌ |
| `--silent` | 禁用日志输出 | ❌ |

## 📁 目录结构

```
prd/
├── prd_from_image_folders.py    # 主评估脚本
├── inception_torch.py           # Inception V3 模型
├── prd_score.py                 # PRD 计算核心
├── inception_v3.pth             # 预训练权重（首次自动下载）
├── prd_cache/                   # 缓存目录（自动创建）
└── README.md                     # 本文档
```

## ⚙️ Inception V3 权重

### 自动下载

首次使用时，如果本地没有权重文件，系统会自动下载：
- **来源**: PyTorch 官方服务器
- **大小**: ~104 MB
- **位置**: `evaluators/prd/inception_v3.pth`

### 手动下载

如果自动下载失败，可手动下载：
```
https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
```

放置在 `evaluators/prd/inception_v3.pth`

## 📊 结果说明

脚本会输出：
- **F_8**: F 分数 (β=8)，更重视召回率
- **F_1/8**: F 分数 (β=1/8)，更重视精确率
- **图表**: PRD 曲线（可选保存为 PNG/PDF）

## 🐛 常见问题

### 图像数量不匹配
```
ValueError: The number of points in eval_data 50 is not equal to...
```
**解决**: 确保真实图像和生成图像数量相同，或编辑脚本添加 `enforce_balance=False`

### 权重下载失败
1. 检查网络连接
2. 手动下载权重文件
3. 放置在 `prd/inception_v3.pth`

### GPU 内存不足
```bash
python prd_from_image_folders.py ... --device cpu
```

### 缓存问题
删除 `prd_cache/` 目录重新计算：
```bash
rm -rf prd_cache/
```

## 📈 性能指标

| 操作 | 时间 | 说明 |
|------|------|------|
| 首次下载权重 | 5-10 分钟 | 网络依赖 |
| 计算 Embeddings | 取决于图像数量 | 可缓存 |
| 计算 PRD | 1-5 分钟 | 取决于样本数 |

## 💡 提示

- **缓存利用**: Embeddings 会被缓存，重复运行更快
- **批处理**: 支持同时评估多个模型
- **GPU 优化**: 自动检测 CUDA，若不可用则自动使用 CPU
- **文档清晰**: 启用 `--verbose`（默认）查看详细日志

## 📝 代码示例

### Python 直接调用

```python
import inception_torch
import numpy as np

# 加载模型
model = inception_torch.InceptionV3FeatureExtractor()

# 假设有图像数据 (numpy array)
# 格式: [batch_size, height, width, 3] 或 [batch_size, 3, height, width]
# 像素范围: [0, 255]

images = np.random.randint(0, 255, (32, 256, 256, 3), dtype=np.uint8)

# 提取特征（自动预处理）
features = model(torch.from_numpy(images).float())
print(features.shape)  # [32, 2048]
```

### 使用嵌入函数

```python
import inception_torch

imgs = np.random.randint(0, 255, (100, 256, 256, 3), dtype=np.uint8)

embeddings = inception_torch.embed_images_in_inception(
    imgs=imgs,
    inception_path='inception_v3.pth',
    layer_name='avg_pool',
    batch_size=32
)

print(embeddings.shape)  # [100, 2048]
```

## 🔗 相关资源

- [PyTorch Inception V3](https://pytorch.org/vision/stable/models/inception.html)
- [PRD 论文](https://arxiv.org/abs/1807.04975)
- [TensorFlow 原始实现](https://github.com/msajjadi/precision-recall-distributions)

## 📄 版本信息

- **版本**: 2.0
- **最后更新**: 2026-03-16
- **Python**: 3.7+
- **依赖**: torch, torchvision, numpy, opencv-python, matplotlib
