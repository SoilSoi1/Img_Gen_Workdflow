# LDM 训练-评估自动化工作流（Agent 执行版）

> 本文档面向 AI Agent，定义从训练到评估的标准操作流程（SOP）。
> 每次执行前必须向用户确认参数，不可擅自决定。

---

## 工作流入口

**触发条件**：用户表达以下任一意图时，激活本工作流
- "开始训练"
- "训练 XXX 类别"
- "评估生成质量"
- "生成图片看看效果"
- "继续训练"
- "训完了，看看结果"

---

## Phase 0: 参数确认（强制）

**规则**：每次进入工作流，必须先向用户确认以下参数。不可使用默认值直接执行。

### 必问参数清单

```
[工作流启动确认]
请确认本次训练的参数：

1. 训练类别（单选）：
   - [ ] leak
   - [ ] tight
   - [ ] 其他：_____

2. 图像尺寸：
   - [ ] 512×512（推荐，当前默认）
   - [ ] 256×256
   - [ ] 其他：_____

3. 训练 epoch 数：
   - [ ] 100（快速验证）
   - [ ] 500（中等）
   - [ ] 1000（完整训练，推荐）
   - [ ] 其他：_____

4. batch_size：
   - [ ] 4（当前默认，5090 显存充裕）
   - [ ] 2（显存紧张时）
   - [ ] 8（追求速度）

5. 是否从头训练还是续训：
   - [ ] 从头训练
   - [ ] 从 checkpoint 续训：_____

6. 本次目标：
   - [ ] 训练 + 自动生成评估报告
   - [ ] 仅训练，评估延后
   - [ ] 仅推理/生成图片（不训练）
```

**用户回复格式**：可以直接说"leak，1000 epoch，batch 4，训练完自动评估"

---

## Phase 1: 环境检查

**执行前自检**：
1. 检查 CUDA 可用：`torch.cuda.is_available()`
2. 检查 VAE 权重存在：`weights/sd-vae-ft-mse/diffusion_pytorch_model.safetensors`
3. 检查数据目录存在：`color_20260321/train/{类别}/`
4. 检查是否有其他 screen 训练在跑（避免显存冲突）

**异常处理**：
- VAE 缺失 → 通过 `hf-mirror.com` 自动下载
- 数据缺失 → 报错并告知用户
- 显存不足 → 建议减小 batch_size 或 image_size

---

## Phase 2: 启动训练

**执行命令模板**：
```bash
screen -dmS {类别}_train bash -c 'cd /root/autodl-tmp/Img_Gen_Workdflow && python3 models/diffusion/ldm/train.py \
    --train_root color_20260321/train/{类别} \
    --image_size {尺寸} --epochs {epoch} --batch_size {bs} \
    --num_workers 4 --save_dir ./experiments/ldm/{类别}_{epoch}ep \
    --ckpt_interval {max(10, epoch//10)} --log_interval 50 --device cuda:0'
```

**启动后动作**：
1. 记录训练启动时间
2. 向用户汇报：`已启动 {类别} 训练，{epoch} epoch，screen 会话：{类别}_train`
3. 告知查看方式：`screen -r {类别}_train` 或 `tail -f experiments/ldm/{类别}_{epoch}ep/training.log`

---

## Phase 3: 训练监控（自动轮询）

**监控策略**：
- 每 30 分钟检查一次训练进度
- 检查方式：读取最新 checkpoint 的 epoch 数，或解析 training.log 的最后几行

**汇报模板**：
```
[{类别}] 训练进度：Epoch {current}/{total}，当前 loss: {loss:.4f}，预计剩余: {eta}
```

**异常检测**：
- loss 连续 50 个 epoch 不下降 → 提醒用户可能需要调参
- loss 突增/NaN → 立即报告，建议检查学习率或数据
- screen 进程消失 → 报告训练中断

---

## Phase 4: 训练结束自动触发

**触发条件**：检测到 `epoch_{total:05d}.pt` 存在，且 screen 进程已结束

**自动执行序列**（无需用户确认，按顺序执行）：

### 4.1 批量推理
```bash
mkdir -p experiments/ldm/{类别}_{epoch}ep/generated_images
python3 models/diffusion/ldm/infer.py \
    --ckpt experiments/ldm/{类别}_{epoch}ep/checkpoints/epoch_{total:05d}.pt \
    --num_samples 200 --ddim_steps 50 --eta 0.0 \
    --output_dir experiments/ldm/{类别}_{epoch}ep/generated_images \
    --device cuda:0
```

### 4.2 运行评估
```python
from evaluators.pipeline import EvaluationPipeline
import json

pipe = EvaluationPipeline(device='cuda')
results = pipe.run(
    gen_dir=f'experiments/ldm/{类别}_{epoch}ep/generated_images',
    real_dir=f'color_20260321/train/{类别}',
    methods=['fid', 'kid', 'lpips', 'brisque'],
    lpips_sample_pairs=1000,
    verbose=True
)

# 保存
with open(f'experiments/ldm/{类别}_{epoch}ep/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 4.3 记录结果到 biye1.md

评估完成后，将结果以**表格行**的形式追加到 `biye1.md` 末尾的实验结果章节。

**追加格式**：

```markdown
### {日期} {类别}_{epoch}ep 评估结果

| 指标 | 数值 | 备注 |
|------|------|------|
| FID | {fid:.2f} | 越低越好（vs 真实集）|
| KID | {kid:.4f} | 越低越好（vs 真实集）|
| LPIPS (intra) | {lpips:.4f} | 生成集内部多样性，越高越好 |
| BRISQUE | {brisque:.2f} | 无参考清晰度，越高越好 |
| 最终 loss | {final_loss:.4f} | 训练收敛值 |
| 训练 epoch | {epoch} | 总训练轮数 |
| batch_size | {bs} | 训练批次 |
| 生成数量 | 200 | 评估用生成图数量 |
| 权重文件 | epoch_{total:05d}.pt | 评估所用 checkpoint |
```

**规则**：
- 每次评估单独一个三级标题（`### 日期 类别_epoch 评估结果`）
- 日期格式：`2026-04-23`
- 不覆盖历史记录，只追加
- 如 `biye1.md` 中尚无"实验结果"章节，自动在文件末尾创建

### 4.4 向用户汇报
```
🎉 {类别} 训练完成！评估结果已追加到 biye1.md。

| 指标 | 数值 |
|------|------|
| FID | {fid:.2f} |
| KID | {kid:.4f} |
| LPIPS | {lpips:.4f} |
| BRISQUE | {brisque:.2f} |

生成图片：experiments/ldm/{类别}_{epoch}ep/generated_images/
评估记录：biye1.md（{日期} {类别}_{epoch}ep 评估结果）

下一步可选：
1. 开始训练另一个类别（tight / leak）
2. 加训更多 epoch
3. 调整参数重新训练
```

---

## Phase 5: 人工审查与决策

**规则**：Phase 4 执行完后，必须等待用户指令，不可自动进入下一轮训练。

**用户可能的选择**：
- "开始训练 tight" → 回到 Phase 0，询问 tight 参数
- "加训 500 epoch" → 确认续训参数，回到 Phase 2
- "生成几张看看" → 执行快速推理（num_samples=4~10）
- "把结果写进论文" → 整理报告格式，追加到 biye1.md
- "暂停，我先看看图" → 停止自动执行，等待用户反馈

---

## 异常处理流程

| 异常 | 检测方式 | 处理动作 |
|------|---------|---------|
| 训练 OOM | nvidia-smi 显示显存满 | 建议 batch_size=2，或 image_size=256 |
| FID/KID 下载 Inception 失败 | 网络超时 | `source /etc/network_turbo` 后重试 |
| checkpoint 损坏 | torch.load 失败 | 尝试加载上一个 epoch（如 epoch_00900.pt） |
| 生成图全黑/全白 | 像素统计异常 | 报告 VAE 解码可能出错，建议检查 |
| tight 数据太少（180张） | 文件计数 | 提醒用户可能过拟合，建议加数据增强 |

---

## 技能化建议（待用户确认）

如需将本工作流固化为 Kimi Skill：
1. 在 `skills/ldm-training/` 下创建 `SKILL.md`
2. 定义触发词：`@ldm-train`、`开始扩散训练`、`训练 leak/tight`
3. 固化 Phase 0 的参数询问模板
4. 绑定本工作流的所有执行命令

用户确认后，我可以立即创建 Skill。
