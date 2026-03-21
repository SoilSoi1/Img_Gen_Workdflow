# DDPM (Denoising Diffusion Probabilistic Models) 实现

该模块实现了DDPM扩散模型用于图像生成任务，基于论文 *Denoising Diffusion Probabilistic Models*。该实现支持图像的训练和采样，特别针对512x512分辨率的灰度图像。

## 📋 模块结构

### 核心模块

#### 1. **Diffusion.py** - 扩散过程实现
包含两个核心类：

- **`GaussianDiffusionTrainer`**（训练时使用）
  - 前向扩散过程：$q(x_t | x_0) = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$
  - 计算预测噪声与真实噪声的MSE损失
  - 参数：
    - `model`：UNet神经网络
    - `beta_1`, `beta_T`：方差计划的起始和结束值
    - `T`：时间步数（总扩散步数）

- **`GaussianDiffusionSampler`**（推理时使用）
  - 反向扩散过程：从纯噪声逐步生成图像
  - 计算后验分布 $p(x_{t-1} | x_t)$
  - 包含采样算法2的完整实现

**关键函数：**
- `extract()`：从1D系列中提取时间步对应的值并reshape用于广播

---

#### 2. **Model.py** - UNet架构

实现了条件U-Net模型，接收噪声图像和时间步，预测添加的噪声。

**核心组件：**

- **`TimeEmbedding`**：时间步嵌入
  - 使用正弦位置编码
  - 输入：时间步索引 $t \in [0, T)$
  - 输出：时间向量（维度=4×通道数）

- **`Swish`**：自门控激活函数 $x \cdot \sigma(x)$

- **`ResBlock`**：残差块
  - 包含两个卷积层
  - 融合时间嵌入信息
  - 可选自注意力层
  - 跳跃连接

- **`AttnBlock`**：多头自注意力块
  - 基于点积注意力
  - 用于捕捉长距离依赖

- **`DownSample` / `UpSample`**：空间分辨率变化
  - DownSample：步长为2的卷积（降采样）
  - UpSample：最近邻插值+卷积（上采样）

- **`UNet`**：主网络架构
  - 编码器（下采样路径）
  - 瓶颈层（中间块）
  - 解码器（上采样路径，包含跳跃连接）
  - 配置参数：
    - `T`：时间步数
    - `ch`：基础通道数
    - `ch_mult`：每层通道倍数 （如 [1, 2, 3, 4] 表示4个下采样阶段）
    - `attn`：应用自注意力的阶段索引
    - `num_res_blocks`：每阶段的残差块数
    - `dropout`：dropout比率
    - `input_channel`：输入通道数（默认3用于RGB图）

---

#### 3. **Scheduler.py** - 学习率调度

- **`GradualWarmupScheduler`**：预热+余弦退火调度器
  - 前N个epoch：线性预热
  - 之后：可选的后续调度器（如余弦退火）
  - 参数：
    - `multiplier`：最终学习率倍数
    - `warm_epoch`：预热步数
    - `after_scheduler`：预热后使用的调度器

---

#### 4. **dataset.py** - 数据加载

- **`LowTimesDataset`**：自定义Dataset类
  - 加载存储在 `tight/` 和 `leak/` 子文件夹的图像
  - 类标签映射：`tight`→0，`leak`→1
  - **多路输出**（用于多模态融合）：
    - `img_resnet`：512×512灰度图（ResNet路径）
      - 随机旋转、翻转、中心裁剪
      - 灰度化、ToTensor、Normalize([0.5], [0.5])
    - `img_vit`：224×224 RGB图（ViT路径）
      - 随机裁剪、ToTensor、ImageNet Normalize
    - `label`：类标签张量

- **`get_dataloader()`**：处理器函数
  - 创建训练和验证DataLoader
  - 返回：`(train_loader, val_loader)`
  - 参数：
    - `train_root`, `val_root`：数据路径
    - `batch_size`, `num_workers`, `pin_memory`：DataLoader配置

---

#### 5. **Train.py** - 训练与评估

- **`train()`**：训练函数
  - 加载数据并初始化模型
  - **优化器**：AdamW（lr=2e-5, weight_decay=1e-4）
  - **学习率调度**：
    1. 预热调度器（前epoch/10步）
    2. 余弦退火
  - **训练流程**：
    - 随机采样时间步
    - 前向扩散添加噪声
    - UNet预测噪声
    - MSE损失反向传播
    - 梯度裁剪（最大范数=1）
  - **检查点保存**：
    - 每个epoch保存最新权重 `last_ckpt.pt`
    - 每500个epoch保存一次版本标记的权重
    - 记录epoch号到文本文件

- **`eval()`**：评估/采样函数
  - 加载训练好的权重
  - 初始化GaussianDiffusionSampler
  - 从标准正态分布采样初始噪声
  - 逐步去噪生成512×512的图像
  - 返回生成的图像张量

---

#### 6. **Main.py** - 配置文件

定义全局配置字典 `modelConfig`，包含所有超参数：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `state` | "eval" | "train"或"eval" |
| `epoch` | 2000 | 训练epoch数 |
| `batch_size` | 2 | 批大小 |
| `T` | 400 | 扩散时间步数 |
| `channel` | 64 | 基础通道数 |
| `channel_mult` | [1,2,3,4] | 各阶段通道倍数 |
| `attn` | [2] | 应用自注意力的阶段 |
| `num_res_blocks` | 1 | 每阶段残差块数 |
| `dropout` | 0.15 | Dropout概率(训练中) |
| `lr` | 2e-5 | 学习率 |
| `multiplier` | 2 | 预热乘数 |
| `beta_1` / `beta_T` | 1e-4 / 0.04 | 方差计划范围 |
| `img_size` | 512 | 输入图像大小 |
| `grad_clip` | 1.0 | 梯度裁剪最大范数 |
| `device` | "cuda:0" | 设备选择 |
| `training_load_weight` | "last_ckpt.pt" | 加载的初始权重 |
| `save_weight_dir` | "./Checkpoints/" | 权重保存目录 |
| `test_load_weight` | "last_ckpt.pt" | 评估时加载的权重 |
| `sampled_dir` | "./SampledImgs/" | 采样图像保存目录 |
| `input_channel` | 1 | 输入通道数（灰度图）|
| `nrow` | 8 | 保存时的网格列数 |

---

#### 7. **gen_quick.py** - 快速采样脚本

- 批量生成样本图像
- **`sampling()`**：采样函数
  - 参数：
    - `num_pic`：生成的图像数量
    - `saved_dir`：保存目录
  - 集成eval()函数逐张生成并保存
- 自动创建输出目录
- 计算并打印总采样时间

---

## 🚀 使用方法

### 要求
```bash
torch>=1.9.0
torchvision
numpy
PIL
tqdm
clean-fid  # 可选，用于FID评估
```

### 关键修改说明

本实现基于 [DDPM-PyTorch](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/) 做了以下重要修改：

#### 1. **输入通道数配置**（RGB三通道）
- 在 `Main.py` 的 `modelConfig` 中添加了 `input_channel` 参数
- 在 `Model.py` 的 `UNet` 类中适配了可变的输入通道数
- 使用RGB图象（`input_channel=3`）

#### 2. **数据归一化修正**（👈 **最关键！**）
**问题**：原始实现使用ImageNet归一化 `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` 会导致像素值范围为 `[-1.996, 2.449]`，与扩散模型所需的 `[-1, 1]` 范围不符，造成SNR不匹配。

**解决方案**：在 `dataset.py` 中修改为：
```python
transforms.ToTensor(),  # [0,1]
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB → [-1,1]
```

**为什么是 [-1,1]？**
- 扩散模型要求：数据均值为0，方差为 $O(1)$
- `[-1,1]` 映射后方差约为 $\frac{1}{3}$，是工程上的最佳实践
- 与GAN、U-Net等模型的标准约定保持一致

#### 3. **检查点管理**
- 每个epoch保存 `last_ckpt.pt`（用于断点续训）
- 每500个epoch保存版本化权重：`ckpt_50k.pt`, `ckpt_100k.pt` 等
- 自动记录epoch号至 `output.txt` 便于追踪

#### 4. **快速采样脚本**
- `gen_quick.py`：批量生成指定数量的图像并自动计时
- 返回采样结果而非直接保存，便于后处理

---

### 训练

#### 基础训练
```python
from Main import modelConfig, main

# 修改配置
modelConfig["state"] = "train"
modelConfig["epoch"] = 2000
modelConfig["device"] = "cuda:0"  # ⚠️ 必须有GPU

# 运行训练
main(modelConfig)
```

#### 数据目录结构
```
newest_data/
├── train/
│   ├── tight/
│   │   ├── img1.png
│   │   ├── img2.png
│   │   └── ...
│   └── leak/
│       ├── img1.png
│       └── ...
└── val/
    ├── tight/
    └── leak/
```

#### 重要配置说明

| 参数 | 说明 | 建议值 |
|-----|------|---------|
| `input_channel` | 输入通道数 | **3**（RGB）|
| `img_size` | 输入分辨率 | 512×512 |
| `batch_size` | 批大小（受显存限制） | 2-4 |
| `T` | 扩散时间步数 | 400（平衡质量与速度）|
| `channel` | 基础通道数 | 64 |
| `beta_1 / beta_T` | 方差计划范围 | 1e-4 / 0.04 |
| `lr` | 学习率 | 2e-5 |
| `grad_clip` | 梯度裁剪 | 1.0 |

#### 断点续训
```python
# 自动加载最新权重继续训练
modelConfig["training_load_weight"] = "last_ckpt.pt"
main(modelConfig)

# 查看已训练的进度
with open("./Checkpoints/output.txt", "r") as f:
    last_epoch = int(f.readlines()[-1])
    print(f"上次训练到第 {last_epoch} epoch")
```

---

### 评估与采样

#### 快速生成样本
```python
from Main import modelConfig
from gen_quick import sampling

# 配置
modelConfig["state"] = "eval"
modelConfig["test_load_weight"] = "ckpt_100k.pt"  # 指定权重
modelConfig["device"] = "cuda:0"

# 生成100张图像，自动计时
sampling(num_pic=100, saved_dir="./outfig/ddpm/results/")
```

#### 命令行方式
```bash
# 直接运行，生成100张图像
python gen_quick.py
```

输出示例：
```
采样 100 张图像总共用时: 45.32 秒
```

#### 自定义采样代码
```python
from Main import modelConfig
from Train import eval
import os

modelConfig["state"] = "eval"
modelConfig["test_load_weight"] = "ckpt_100k.pt"

# 单张采样
sampled_img = eval(modelConfig)  # 返回 tensor，shape: [1, 1, 512, 512]

# 保存
from torchvision.utils import save_image
save_image(sampled_img, "output.png")
```

---

### 质量评估（FID/KID）

```python
from evaluators._fid import cal_fid
from evaluators._kid import cal_kid

# 计算FID分数
fid_score = cal_fid(
    path_real="./newest_data/val/tight/",
    path_fake="./outfig/ddpm/results/",
    device="cuda:0"
)
print(f"FID Score: {fid_score:.2f}")

# 计算KID分数
kid_score = cal_kid(
    path_real="./newest_data/val/tight/",
    path_fake="./outfig/ddpm/results/",
    device="cuda:0"
)
print(f"KID Score: {kid_score:.4f}")
```

---

### 显存优化技巧

如果遇到 CUDA 显存不足：

```python
# 减小批大小
modelConfig["batch_size"] = 1

# 减少通道数（会影响质量）
modelConfig["channel"] = 32
modelConfig["channel_mult"] = [1, 2, 2]  # 减少层数

# 降低分辨率（作为最后手段）
modelConfig["img_size"] = 256
```

---

## ⚡ 快速训练脚本 (train_quick.py)

### 📌 概述

`train_quick.py` 是一个便捷的训练脚本，让你通过 **迭代次数** 而不是 epoch 来定义训练量。脚本会自动根据数据集大小计算所需的 epoch 数。

### 🎯 核心特性

1. **基于迭代次数的训练量定义** - 指定总迭代次数，脚本自动计算所需 epoch 数
2. **灵活的 Checkpoints 管理** - 自定义文件夹名称 + 自动时间戳 + 灵活保存间隔
3. **优雅的训练日志** - JSON 格式，记录每 epoch 的 loss、学习率、迭代数、时间戳
4. **断点续训支持** - 轻松从中断位置恢复训练
5. **干运行模式** - `--dry_run` 仅检查配置不训练

### 🚀 快速开始

#### 最简单的使用（默认参数）
```bash
python train_quick.py --total_iterations 50000
```

这会自动：
- 读取 `./newest_data/train/` 中的数据
- 计算每 epoch 的迭代次数（数据量 / batch_size）
- 计算所需 epoch 数
- 在 `./Checkpoints/{timestamp}/` 下保存权重和日志

#### 自定义 Checkpoints 名称
```bash
python train_quick.py --total_iterations 50000 --ckpt_name "exp_1"
```
会创建：`./Checkpoints/exp_1/{timestamp}/`

#### 设置保存间隔
```bash
python train_quick.py \
  --total_iterations 50000 \
  --ckpt_name "exp_1" \
  --ckpt_interval 5000
```
- 每 5,000 次迭代保存一次权重
- 脚本自动转换为 epoch 间隔

#### 自定义数据路径
```bash
python train_quick.py \
  --total_iterations 50000 \
  --train_root /path/to/training/data \
  --val_root /path/to/validation/data \
  --ckpt_name "custom_exp"
```

#### 断点续训
```bash
python train_quick.py \
  --total_iterations 100000 \
  --ckpt_name "exp_1" \
  --resume last_ckpt.pt
```

#### 仅检查配置（不训练）
```bash
python train_quick.py \
  --total_iterations 50000 \
  --train_root /path/to/data \
  --dry_run  # 只打印配置，不开始训练
```

### 📊 输出示例
```
======================================================================
🚀 DDPM 快速训练脚本
======================================================================
✓ 训练数据: ./newest_data/train/
✓ 验证数据: ./newest_data/val/
👉 每个 epoch 的迭代次数: 500
   (数据集: 1000 张图 / 批大小: 2)

📊 迭代计划:
   目标总迭代次数: 50,000
   所需 epoch 数:  100
   实际总迭代次数: 50,000 (比目标多 0)

💾 Checkpoints 保存位置: ./Checkpoints/exp_1/20260321_143022

💾 检查点保存间隔:
   每 5,000 迭代保存一次
   ≈ 每 10 epoch 保存一次

⚙️  训练参数:
   学习率: 0.00002
   总 batch: 2
   T (扩散步数): 400
   设备: cuda:0

📋 配置摘要:
   状态: train
   总 epoch 数: 100
   日志将保存到: ./Checkpoints/exp_1/20260321_143022/training_log_*.json

⏱️  开始训练...
```

### 📝 命令行参数完整列表

#### 迭代次数相关
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--total_iterations` | int | 50000 | 总迭代次数 |
| `--batch_size` | int | 2 | 批大小 |

#### 数据集路径
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_root` | str | `./newest_data/train/` | 训练数据目录 |
| `--val_root` | str | `./newest_data/val/` | 验证数据目录 |

#### Checkpoints 配置
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ckpt_root` | str | `./Checkpoints/` | Checkpoints 根目录 |
| `--ckpt_name` | str | None | 自定义文件夹名称（可选） |
| `--ckpt_interval` | int | 10000 | 保存间隔（迭代数，0=仅保存最后的） |

#### 其他参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--resume` | str | None | 指定断点续训的权重文件 |
| `--dry_run` | bool | False | 仅检查配置，不训练 |

### 📂 文件结构

训练后的输出结构：
```
Checkpoints/
└── exp_1/
    └── 20260321_143022/                    # 自动时间戳
        ├── training_log_20260321_143022.json   # JSON 日志
        ├── last_ckpt.pt                    # 最新权重
        ├── ckpt_10_epoch.pt                # 版本化权重
        └── ...
```

### 📋 训练日志格式 (JSON)

```json
{
  "start_time": "2026-03-21T14:30:22.123456",
  "config": {
    "lr": 0.00002,
    "batch_size": 2,
    "T": 400,
    "train_root": "./newest_data/train/"
  },
  "history": [
    {
      "epoch": 0,
      "loss": 0.054321,
      "lr": 4e-6,
      "iterations_in_epoch": 500,
      "timestamp": "2026-03-21T14:30:45.654321"
    },
    {
      "epoch": 1,
      "loss": 0.043210,
      "lr": 8e-6,
      "iterations_in_epoch": 500,
      "timestamp": "2026-03-21T14:31:30.123456"
    }
  ],
  "last_epoch": 1,
  "last_update_time": "2026-03-21T14:31:30.123456"
}
```

### 🔄 迭代次数计算原理

```
每 epoch 的迭代次数 = ⌈ 数据集大小 / batch_size ⌉

所需 epoch 数 = ⌈ 总迭代次数 / 每 epoch 迭代数 ⌉

实际总迭代次数 = 所需 epoch 数 × 每 epoch 迭代数
```

例如：
- 数据集: 1000 张图
- Batch size: 2
- 每 epoch 迭代数: ⌈1000 / 2⌉ = 500
- 目标迭代: 50,000
- 所需 epoch: ⌈50,000 / 500⌉ = 100
- 实际迭代: 100 × 500 = 50,000

### ⚠️ 常见问题

**Q1: 如何知道数据集的大小？**  
脚本会自动计算。运行时会打印：
```
👉 每个 epoch 的迭代次数: 500
   (数据集: 1000 张图 / 批大小: 2)
```

**Q2: Checkpoints 如何命名？**  
- 指定 `--ckpt_name "exp_1"`：`./Checkpoints/exp_1/{timestamp}/`
- 不指定：`./Checkpoints/{timestamp}/`

时间戳格式：`YYYYMMDD_HHMMSS`

**Q3: 可以改变保存间隔吗？**  
用 `--ckpt_interval` 指定迭代数，脚本自动转换为 epoch 间隔。
例如 `--ckpt_interval 5000` 表示每 5,000 次迭代保存一次。

**Q4: 日志文件在哪里？**  
在 checkpoints 目录下，文件名：`training_log_{timestamp}.json`

例如：`./Checkpoints/exp_1/20260321_143022/training_log_20260321_143022.json`

### 💡 快速参考

| 场景 | 命令 |
|------|------|
| 快速测试 | `python train_quick.py --total_iterations 1000 --dry_run` |
| 标准训练 | `python train_quick.py --total_iterations 50000 --ckpt_name "exp_1"` |
| 长期训练 | `python train_quick.py --total_iterations 200000 --ckpt_name "long_exp" --ckpt_interval 10000` |
| 断点续训 | `python train_quick.py --total_iterations 100000 --ckpt_name "exp_1" --resume last_ckpt.pt` |

---

## 📊 模型架构详解

```
输入: x_t (512×512), t (时间步)
  ↓
时间嵌入 (TimeEmbedding)
  ↓
下采样块 (DownSample Paths)
  ├─ 残差块 + 可选自注意力
  ├─ 卷积降采样
  └─ 记录特征用于跳跃连接
  ↓
中间块 (Middle Blocks)
  ├─ 残差块 + 自注意力
  └─ 残差块
  ↓
上采样块 (UpSample Paths)
  ├─ 连接下采样对应特征
  ├─ 残差块 + 可选自注意力
  └─ 最近邻上采样
  ↓
输出卷积 (Tail Conv)
  ↓
输出: 预测噪声 (512×512)
```

---

## 🔄 扩散过程原理

### 前向过程（训练）
给定原始图像 $x_0$，在时间步 $t$ 添加高斯噪声：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0,I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} (1-\beta_s)$

### 反向过程（采样）
从纯噪声 $x_T \sim \mathcal{N}(0,I)$ 开始，逐步去噪：
$$x_{t-1} \sim p(x_{t-1}|x_t) = \mathcal{N}(\mu_t, \sigma_t^2)$$

其中 $\mu_t$ 由UNet预测的噪声计算得出。

---

## ⚙️ 关键配置优化建议

| 场景 | 推荐调整 |
|------|---------|
| 显存不足 | ↓batch_size, ↓channel, ↓ch_mult长度 |
| 训练效果差 | ↑lr或调整schedule, 增加warmup_epoch |
| 采样速度慢 | ↓T（时间步数） |
| 生成质量差 | ↑num_res_blocks, 加在更多层的attn |
| 低分辨率 | 修改img_size和网络配置 |

---

## 📁 文件依赖关系

```
Main.py (配置) 
  └─> Train.py (训练/推理)
      ├─> Diffusion.py (扩散过程)
      ├─> Model.py (UNet架构)
      ├─> Scheduler.py (学习率调度)
      └─> dataset.py (数据加载)
  
gen_quick.py (采样脚本)
  └─> Main.py & Train.py
```

---

## 💡 扩展建议

1. **条件生成**：修改UNet以支持类别条件
2. **更高分辨率**：调整通道倍数和时间步数
3. **多GPU训练**：使用 `DataParallel` 或 `DistributedDataParallel`
4. **EMA权重**：为模型参数添加指数移动平均
5. **FID/IS评估**：集成评估指标

---

## 📝 输出示例

- **训练输出**：
  ```
  Epoch: 0, Loss: 0.0234, LR: 4e-6
  Epoch: 1, Loss: 0.0198, LR: 8e-6
  ...
  ```

- **采样输出**：生成的图像保存在 `./outfig/ddpm/<weight>/sampled_img_*.png`

---

## 🔗 参考文献

- **DDPM论文**：Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- **官方实现**：github.com/hojonathanho/diffusion

---

**最后更新**：2026年3月
