# 图像生成质量评估管道 (EvaluationPipeline)

统筹调用多种图像质量评估方法的综合管道。支持单数据集和双数据集评估方法，并能灵活配置各个评估器的参数。

## 功能特性

- **单数据集方法**：无需参考图像，直接评估生成图像集质量
  - LPIPS：感知相似度评估（支持模型选择和采样）
  - BRISQUE：无参考图像质量评估

- **双数据集方法**：对比生成图像与真实图像
  - FID：Fréchet Inception Distance
  - KID：Kernel Inception Distance（无偏估计）
  - PRD：Precision & Recall for Distribution（可选）

- **灵活配置**
  - 自由组合评估方法
  - 自动参数验证（缺少必需参数时会跳过）
  - 支持CPU和GPU计算
  - 结果导出为JSON格式

## 快速开始

### 作为Python函数使用

```python
from evaluators.pipeline import EvaluationPipeline

pipeline = EvaluationPipeline(device='cuda')
results = pipeline.run(
    gen_dir='./path/to/generated',
    real_dir='./path/to/real',
    methods=['fid', 'kid', 'lpips', 'brisque']
)
print(results)
# 输出: {'fid': 395.13, 'kid': 0.357, 'lpips': 0.366, 'brisque': 31.71}
```

### 命令行使用

```bash
# 计算所有可用指标
python evaluators/pipeline.py \
  --gen_dir ./path/to/generated \
  --real_dir ./path/to/real \
  --methods fid,kid,lpips,brisque \
  --device cuda \
  --output json

# 仅计算单数据集方法
python evaluators/pipeline.py \
  --gen_dir ./path/to/generated \
  --methods lpips,brisque \
  --lpips_net vgg

# 使用采样加快LPIPS计算
python evaluators/pipeline.py \
  --gen_dir ./path/to/generated \
  --methods lpips \
  --lpips_sample_pairs 1000
```

## 参数说明

### 必需参数
- `--gen_dir`：生成图像所在文件夹

### 可选参数
- `--real_dir`：真实图像所在文件夹（双数据集方法需要）
- `--methods`：要执行的评估方法（默认: fid,kid,lpips,brisque）
  - 单数据集：`lpips,brisque`
  - 双数据集：`fid,kid,prd`
- `--device`：计算设备 `cuda` 或 `cpu`（默认自动选择）
- `--lpips_net`：LPIPS网络类型 `alex|vgg|squeeze`（默认: alex）
- `--lpips_sample_pairs`：LPIPS采样对数（默认计算全部）
- `--prd_inception_path`：PRD的Inception模型路径
- `--output`：输出格式 `json` 或 `dict`

## 方法介绍

### LPIPS (Learned Perceptual Image Patch Similarity)
- **输入**：单个图像数据集
- **计算方式**：计算数据集内所有图像对的感知相似度，取平均值
- **分数解释**：值越小表示多样性越好，值越大表示样本相似（可能有模式坍缺）
- **网络选择**：
  - `alex`（默认）：最快，AlexNet特征
  - `vgg`：更精细，VGGNet特征
  - `squeeze`：轻量级，SqueezeNet特征
- **采样选项**：大数据集可用 `--lpips_sample_pairs` 加快计算

### BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- **输入**：单个图像数据集
- **计算方式**：基于自然场景统计评估每张图像质量
- **分数解释**：0-100范围，值越小表示图像质量越好
- **特点**：无需参考图像，计算速度快

### FID (Fréchet Inception Distance)
- **输入**：真实图像数据集 + 生成图像数据集
- **计算方式**：计算两个图像集在Inception特征空间中的Wasserstein距离
- **分数解释**：值越小越好，0表示完全相同
- **应用**：衡量生成图像与真实数据的相似度

### KID (Kernel Inception Distance)
- **输入**：真实图像数据集 + 生成图像数据集
- **计算方式**：基于核方法的无偏估计，使用MMD
- **分数解释**：值越小越好
- **优势**：比FID更稳定的无偏估计

### PRD (Precision & Recall for Distribution)
- **输入**：真实图像数据集 + 生成图像数据集
- **计算方式**：评估生成分布相对于真实分布的精确率和召回率
- **返回值**：F-beta分数（综合指标）
- **注意**：需要额外的Inception模型权重

## 使用建议

### 快速评估（推荐用于调试）
```bash
python evaluators/pipeline.py \
  --gen_dir ./outputs \
  --methods lpips,brisque
```
耗时 < 1 分钟，快速了解生成图像的基本质量。

### 标准评估（推荐用于论文）
```bash
python evaluators/pipeline.py \
  --gen_dir ./outputs \
  --real_dir ./data/real \
  --methods fid,kid,lpips,brisque \
  --device cuda
```
耗时 5-10 分钟，提供全面的质量评估。

### 精细评估（大数据集）
```bash
python evaluators/pipeline.py \
  --gen_dir ./outputs \
  --real_dir ./data/real \
  --methods fid,lpips \
  --lpips_net vgg \
  --lpips_sample_pairs 5000 \
  --device cuda \
  --output json
```

## 返回结果格式

```python
{
    'fid': 395.1278,           # Fréchet Inception Distance
    'kid': 0.3570,             # Kernel Inception Distance
    'lpips': 0.3664,           # LPIPS平均值
    'brisque': 31.7136,        # BRISQUE平均分数
    'prd': {                   # 若计算PRD
        'F-beta': 0.85,
        'precision': 0.88,
        'recall': 0.82
    }
}
```

## 扩展开发

### 添加新的评估方法

在 `pipeline.py` 中添加新方法：

```python
def _eval_custom(self, image_dir: str) -> float:
    """计算自定义指标"""
    try:
        from my_evaluator import calculate_score
        score = calculate_score(image_dir)
        return score
    except Exception as e:
        print(f"计算失败: {e}")
        return None
```

然后在 `run()` 方法中添加调用：

```python
if 'custom' in methods:
    results['custom'] = self._eval_custom(gen_dir)
```

## 常见问题

**Q: 为什么LPIPS计算很慢？**
A: LPIPS需要对所有图像对计算感知相似度。可以用 `--lpips_sample_pairs` 参数进行采样加快。

**Q: FID/KID计算失败，显示"divide by zero"？**
A: 这是数值稳定性问题。在小数据集上很常见，但不影响相对比较。

**Q: 可以在colab上运行吗？**
A: 可以，只需保证依赖库已安装（torch, torchvision, cleanfid, lpips, brisque等）。

**Q: 如何保存评估结果？**
A: 使用 `--output json` 参数，结果会保存为 `evaluation_results.json`。
