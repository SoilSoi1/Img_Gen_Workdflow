# 实验数据路径速查表

> 所有路径均为绝对路径。此文件与 `biye1.md` 同步更新。

---

## 训练数据

```
LEAK 预处理（512×512）:  /root/autodl-tmp/Img_Gen_Workdflow/dataset/LEAK_PROCESSED/    (1003 张)
TIGHT 预处理（512×512）: /root/autodl-tmp/Img_Gen_Workdflow/dataset/TIGHT_PROCESSED/   (209 张)
```

## 阶段一：LR 筛选结果

```
汇总 CSV:    /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/summary.csv

LEAK lr=5e-6:  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/20260507-leak_processed_lr5e-6_ch192/
LEAK lr=1e-5:  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/20260508-leak_processed_lr1e-5_ch192/
LEAK lr=2e-5:  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/20260508-leak_processed_lr2e-5_ch192/

TIGHT lr=5e-6: /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/20260508-tight_processed_lr5e-6_ch192/
TIGHT lr=1e-5: /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/20260508-tight_processed_lr1e-5_ch192/
TIGHT lr=2e-5: /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase1_lr_search/20260508-tight_processed_lr2e-5_ch192/
```

## 阶段二：最佳 epochs 确定结果

```
FID 曲线 CSV:      /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase2_epoch_search/fid_vs_epochs.csv
完整 6 指标 CSV:   /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase2_epoch_search/full_eval_summary.csv

LEAK lr=4e-5  best ep=1400:
  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase2_epoch_search/20260508-leak_processed_lr1e-5_ch192_ep1500/

LEAK lr=8e-5  best ep=800:
  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase2_epoch_search/20260509-leak_processed_lr2e-5_ch192_ep1500/

TIGHT lr=4e-5 best ep=1300:
  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase2_epoch_search/20260510-tight_processed_lr1e-5_ch192_ep1500/

TIGHT lr=8e-5 best ep=1000:
  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase2_epoch_search/20260509-tight_processed_lr2e-5_ch192_ep1500/
```

## 阶段三：模型大小对比结果（正在运行）

```
根目录:  /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase3_model_size/
汇总 CSV: /root/autodl-tmp/Img_Gen_Workdflow/experiments/ldm/phase3_model_size/summary.csv
```

## 核心代码

```
训练:    /root/autodl-tmp/Img_Gen_Workdflow/models/diffusion/ldm/train.py
推理:    /root/autodl-tmp/Img_Gen_Workdflow/models/diffusion/ldm/infer.py
批量推理: /root/autodl-tmp/Img_Gen_Workdflow/models/diffusion/ldm/batch_infer.py
模型定义: /root/autodl-tmp/Img_Gen_Workdflow/models/diffusion/ldm/model.py
数据集:  /root/autodl-tmp/Img_Gen_Workdflow/models/diffusion/ldm/dataset.py
DDIM:   /root/autodl-tmp/Img_Gen_Workdflow/models/diffusion/ldm/ddim.py
```

## 评估脚本

```
FID:     /root/autodl-tmp/Img_Gen_Workdflow/evaluators/_fid.py
KID:     /root/autodl-tmp/Img_Gen_Workdflow/evaluators/_kid.py
LPIPS:   /root/autodl-tmp/Img_Gen_Workdflow/evaluators/lpips_pairwise.py
BRISQUE: /root/autodl-tmp/Img_Gen_Workdflow/evaluators/brisque_official.py
PRD:     /root/autodl-tmp/Img_Gen_Workdflow/evaluators/prd/prd_from_image_folders.py
```

## 预训练权重

```
SD VAE:        /root/autodl-tmp/Img_Gen_Workdflow/weights/sd-vae-ft-mse/
Inception V3:  /root/autodl-tmp/Img_Gen_Workdflow/evaluators/prd/inception_v3.pth
```

## 对比可视化

```
grid_compare.py: /root/autodl-tmp/Img_Gen_Workdflow/grid_compare.py
```
