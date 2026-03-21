import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from Scheduler import GradualWarmupScheduler
from dataset import get_dataloader  # 新增：使用自定义数据集加载器

def epoch_file(number, filename="output.txt"):
    # 确保输入是一个整数
    if not isinstance(number, int):
        # 尝试将其转换为整数，如果失败则抛出错误
        try:
            number = int(number)
        except ValueError:
            print(f"错误: 输入 '{number}' 无法转换为整数。")
            return

    # 使用 'a' 模式打开文件，表示追加 (append)。
    # 如果文件不存在，'a' 模式会自动创建文件。
    try:
        with open(filename, 'a') as file:
            # 将整数转换为字符串，并添加换行符 '\n'，实现逐行保存
            file.write(str(number) + '\n')
        print(f"成功将整数 {number} 保存到文件 '{filename}' 中。")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    # 使用自定义数据集（只取 ResNet 通路的图像）
    # get_dataloader 返回 (train_loader, val_loader)，这里只取 train_loader
    dataloader, _ = get_dataloader(
        train_root="./newest_data/train/",
        val_root="./newest_data/val/",  # 若有单独 val 目录可改为 val 路径
        batch_size=modelConfig["batch_size"],
        num_workers=4,
        pin_memory=True
    )

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            # LowTimesDataset 返回 (img_resnet, img_vit, label)
            for img_resnet, img_vit, labels in tqdmDataLoader:
                 # train
                optimizer.zero_grad()
                # 仅使用 ResNet 通路的图像作为输入
                x_0 = img_resnet.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'last_ckpt.pt'))
        if e % 500 == 0 and e != 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{e}epoch.pt'))
        epoch_file(e, f'{modelConfig["save_weight_dir"]}/output.txt')


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0., input_channel=modelConfig["input_channel"])
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        # 你的数据为 512x512，因此这里使用 512
        noisyImage = torch.randn(
            size=[1, int(modelConfig["input_channel"]), 512, 512], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # save_image(sampledImgs, os.path.join(
        #     modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])

        return sampledImgs