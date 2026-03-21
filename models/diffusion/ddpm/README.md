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
