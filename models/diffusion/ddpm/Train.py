import os
from typing import Dict
from datetime import datetime
import json

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
from dataset import get_dataloader


def create_training_log_file(save_dir, initial_info=None):
    """
    创建一个带有时间戳和训练信息的日志文件
    
    参数:
        save_dir (str): 日志保存目录
        initial_info (dict): 初始信息（如模型配置）
    
    返回:
        str: 日志文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"training_log_{timestamp}.json")
    
    # 初始化日志数据结构
    log_data = {
        "start_time": datetime.now().isoformat(),
        "config": initial_info or {},
        "history": []  # 存储 epoch 级别的信息
    }
    
    # 写入初始日志
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    return log_file


def log_epoch_info(log_file, epoch, loss, lr, iterations_in_epoch=1):
    """
    记录每个 epoch 的信息到 JSON 日志文件
    
    参数:
        log_file (str): 日志文件路径
        epoch (int): epoch 号
        loss (float): 该 epoch 的平均损失
        lr (float): 当前学习率
        iterations_in_epoch (int): 该 epoch 的迭代次数（关键指标！）
    """
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # 添加当前 epoch 的信息
        epoch_info = {
            "epoch": epoch,
            "loss": round(float(loss), 6),
            "lr": float(lr),
            "iterations_in_epoch": iterations_in_epoch,
            "timestamp": datetime.now().isoformat()
        }
        log_data["history"].append(epoch_info)
        log_data["last_epoch"] = epoch
        log_data["last_update_time"] = datetime.now().isoformat()
        
        # 更新日志文件
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"警告: 写入日志文件失败 {e}，将继续训练")

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    # 使用自定义数据集（只取 ResNet 通路的图像）
    # get_dataloader 返回 (train_loader, val_loader)，这里只取 train_loader
    dataloader, _ = get_dataloader(
        train_root=modelConfig.get("train_root", "./newest_data/train/"),
        val_root=modelConfig.get("val_root", "./newest_data/val/"),
        batch_size=modelConfig["batch_size"],
        num_workers=modelConfig.get("num_workers", 4),
        pin_memory=modelConfig.get("pin_memory", True)
    )
    
    # 创建或获取日志文件
    log_file = modelConfig.get("log_file", None)
    if log_file is None:
        log_file = create_training_log_file(
            modelConfig["save_weight_dir"],
            initial_info={
                "lr": modelConfig["lr"],
                "batch_size": modelConfig["batch_size"],
                "T": modelConfig["T"],
                "train_root": modelConfig.get("train_root", "./newest_data/train/")
            }
        )
        modelConfig["log_file"] = log_file
    
    print(f"日志文件: {log_file}")
    
    # 计算每个 epoch 的迭代次数
    iterations_per_epoch = len(dataloader)

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
        epoch_loss = 0.0
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
                epoch_loss += loss.item()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        
        # 计算该 epoch 的平均损失
        avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
        current_lr = optimizer.state_dict()['param_groups'][0]["lr"]
        
        # 记录到日志文件
        try:
            log_epoch_info(log_file, e, avg_loss, current_lr, iterations_per_epoch)
        except Exception as log_err:
            print(f"警告: 日志记录失败 - {log_err}")
        
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'last_ckpt.pt'))
        if e % 500 == 0 and e != 0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], f'ckpt_{e}epoch.pt'))


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