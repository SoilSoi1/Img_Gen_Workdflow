# Build Instructions
This md file describes the process of building this repository.
## 预期文件结构
```txt
project_root/
│
├── configs/                        # 所有 YAML 配置（模块化、可复用）
│   ├── default.yaml                # 全局默认配置（实验名、seed、device等）
│   ├── dataset.yaml                # 数据集与预处理
│   ├── training.yaml               # 训练相关默认项
│   ├── eval.yaml                   # 评估相关配置
│   ├── env.yaml                    # 环境记录（可选：docker、conda等）
│   └── model/                      # 各类模型自己的配置
│       ├── diffusion.yaml
│       ├── gan.yaml
│       ├── vae.yaml
│       └── <other>.yaml
│
├── models/                         # 所有模型类（统一继承 base_model）
│   ├── base_model.py               # 模型基类（train/eval/gen接口规范）
│   ├── diffusion/                  # Diffusion 实现
│   │   ├── ddpm.py
│   │   ├── unet.py
│   │   └── sampler.py
│   ├── gan/
│   │   ├── dcgan.py
│   │   ├── stylegan.py
│   │   └── discriminator.py
│   ├── vae/
│   │   ├── vae.py
│   │   └── encoder_decoder.py
│   └── utils/                      # 模型层与通用代码
│       ├── layers.py
│       └── ops.py
│
├── data/                           # 原始数据（只读）
│   ├── raw/
│   ├── processed/
│   └── splits/                     # train/val/test json 或 csv
│
├── datasets/                       # Dataset 类（统一接口）
│   ├── base_dataset.py
│   ├── image_folder.py
│   ├── paired_dataset.py
│   └── transforms/
│
├── experiments/                    # 每一次实验自动创建一个子文件夹（极关键）
│   └── exp_YYYYMMDD_HHMMSS/        # 例如 exp_20251119_221045
│       ├── config_used.yaml        # 本次实验真正使用的合并配置（记录可复现性）
│       ├── logs/                   # 日志、事件、训练曲线
│       ├── checkpoints/            # ckpt（按 epoch 命名）
│       ├── samples/                # 所有 sample（按每个 epoch / seed 分类）
│       └── eval/                   # 评估结果（FID/LPIPS/SSIM…）
│           ├── metrics.json
│           ├── diagnostics/        # 图表、最近邻可视化等
│           └── samples_evaled/
│
├── evaluators/                     # 所有评估函数
│   ├── fid.py
│   ├── kid.py
│   ├── lpips.py
│   ├── ssim_psnr.py
│   ├── prd.py                      # Precision-Recall / Density-Coverage
│   └── Evaluator.py                # 统一调度入口（EvalManager）, 评估基类
│
├── pipelines/                      # 工作流与统一入口
│   ├── train.py                    # 主训练入口（自动加载 base_model → 子类）
│   ├── eval.py                     # 一键评估入口
│   ├── sample.py                   # 一键生成入口
│   └── orchestrator.py             #（可选）自动跑多实验矩阵
│
├── utils/
│   ├── config_loader.py            # 合并 YAML；类似 Hydra，但更简单
│   ├── logger.py                   # 日志统一化（tensorboard / wandb）
│   ├── seed.py                     # 固定随机种子
│   ├── visualization.py            # 样本网格、t-SNE等
│   └── fileio.py
│
├── scripts/                        # 命令行脚本或 Slurm 任务
│   ├── run_train.sh
│   ├── run_eval.sh
│   ├── run_sample.sh
│   └── run_matrix.sh
│
├── docs/                           # 全部文档（你要求完整记录实验）
│   ├── experiment_log.md           # 每次实验的总结笔记
│   ├── model_notes/
│   └── architecture_diagrams/
│
└── README.md
```

## *2025年11月19日*
### 设计BaseModel基类
- 统一接口：train(), inference()  
实现基类初步实现：
```python
class BaseModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.model = None # 子类实现具体模型
    def train(self, dataloader):
        ...
        return self
    def inference(self, input_data):
        ...
        return img
    def save_ckpt(self, path):
        ...
    def load_ckpt(self, path):
        ...
    # 需要强制子类实现的方法
    @abstractmethod
    def _train_step(self, dataloader): 
        raise NotImplementedError

    @abstractmethod
    def _inference_step(self, z): 
        raise NotImplementedError
```
