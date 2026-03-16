#!/usr/bin/env python3
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
# PyTorch Implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F


class InceptionV3FeatureExtractor(nn.Module):
    """
    Inception V3 特征提取器
    提取指定层的特征用于评估生成模型的质量
    """
    
    def __init__(self, inception_path=None, layer_name='avg_pool'):
        """
        初始化Inception特征提取器
        
        参数：
            inception_path: 预训练权重文件的路径（.pth格式）
            layer_name: 要提取特征的层名称，支持：
                       - 'avg_pool': 平均池化层输出（默认，维度2048）
                       - 'fc': 全连接层输出（维度1000）
        """
        super(InceptionV3FeatureExtractor, self).__init__()
        
        # 加载预训练的Inception V3模型
        self.inception = models.inception_v3(pretrained=False)
        
        # 如果提供了权重路径，加载自定义权重
        if inception_path and os.path.exists(inception_path):
            self.load_weights(inception_path)
        else:
            print("⚠️  警告: 未加载预训练权重，模型使用随机初始化参数运行!")
        
        # 设置特征提取层
        self.layer_name = layer_name
        self._set_feature_extraction_mode(layer_name)
    
    def load_weights(self, inception_path):
        """
        从文件加载预训练权重，并进行详细的验证
        
        参数：
            inception_path: 权重文件路径（.pth格式）
        """
        if not os.path.exists(inception_path):
            raise ValueError(f'Inception network file not found: {inception_path}')
        
        # 加载权重，允许部分权重不匹配
        checkpoint = torch.load(inception_path, map_location='cpu')
        
        # 处理不同的权重格式
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除 'module.' 前缀（如果是DataParallel保存的模型）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # 加载权重并收集加载信息
        missing_keys, unexpected_keys = self.inception.load_state_dict(new_state_dict, strict=False)
        
        # ============ 权重加载验证 ============
        model_keys = set(self.inception.state_dict().keys())
        loaded_keys = set(new_state_dict.keys())
        matched_keys = model_keys & loaded_keys
        
        print("\n" + "="*70)
        print("权重加载验证信息 (Weight Loading Verification)")
        print("="*70)
        print(f"模型总参数数量: {len(model_keys)}")
        print(f"加载文件中的参数数量: {len(loaded_keys)}")
        print(f"成功匹配的参数数量: {len(matched_keys)}")
        print(f"匹配率: {len(matched_keys) / len(model_keys) * 100:.1f}%")
        
        if missing_keys:
            print(f"\n⚠️  未加载的参数 ({len(missing_keys)} 个):")
            for key in list(missing_keys)[:5]:  # 只显示前5个
                print(f"   - {key}")
            if len(missing_keys) > 5:
                print(f"   ... 还有 {len(missing_keys) - 5} 个")
        
        if unexpected_keys:
            print(f"\n⚠️  加载文件中多余的参数 ({len(unexpected_keys)} 个):")
            for key in list(unexpected_keys)[:5]:  # 只显示前5个
                print(f"   - {key}")
            if len(unexpected_keys) > 5:
                print(f"   ... 还有 {len(unexpected_keys) - 5} 个")
        
        # 关键验证：检查是否加载了足够的权重
        if len(matched_keys) < len(model_keys) * 0.5:
            print("\n❌ 警告：加载的权重不足50%，模型可能使用了大量随机初始化的参数!")
            print("   建议检查权重文件的参数名称格式是否匹配。")
        elif len(matched_keys) > len(model_keys) * 0.8:
            print("\n✓ 权重加载成功：大部分参数已从文件加载")
        
        print("="*70 + "\n")
        
        # 若要强制检查关键参数是否加载，可以取消下面的注释
        # self._verify_critical_weights()
    
    def _verify_critical_weights(self):
        """
        验证关键层的权重是否被成功加载（而非随机初始化）
        通过检查权重的统计特性来判断
        """
        print("\n检查关键层权重统计信息...")
        critical_layers = [
            'Conv2d_1a_3x3.conv.weight',
            'Conv2d_2a_3x3.conv.weight',
            'Mixed_5b.branch1x1.conv.weight',
            'avgpool'
        ]
        
        state_dict = self.inception.state_dict()
        for layer_name in critical_layers:
            if layer_name in state_dict:
                weight = state_dict[layer_name]
                # 随机初始化的权重通常均值接近0，标准差较小
                # 预训练权重通常有更大的方差
                if weight.dim() > 0:
                    mean_val = weight.mean().item()
                    std_val = weight.std().item()
                    print(f"  {layer_name}: mean={mean_val:.6f}, std={std_val:.6f}")
            else:
                print(f"  ⚠️  {layer_name} 不存在于模型中")
    
    def _set_feature_extraction_mode(self, layer_name):
        """
        设置模型为特征提取模式，冻结所有参数并移除分类头
        
        参数：
            layer_name: 要提取特征的层名称
        """
        # 冻结所有参数
        for param in self.inception.parameters():
            param.requires_grad = False
        
        # 根据指定的层设置前向传播钩子
        self.feature_layer = layer_name
        
        # 移除原始的全连接和辅助分类器
        self.inception.fc = nn.Identity()
        self.inception.AuxLogits = None
    
    def forward(self, x):
        """
        前向传播，提取指定层的特征
        
        参数：
            x: 输入张量，形状 [batch_size, 3, 299, 299]
               像素值范围 [-1, 1]（已经过preprocess_for_inception处理）
        
        返回：
            特征张量，形状 [batch_size, feature_dim]
        """
        # 确保模型处于评估模式
        self.eval()
        
        # 执行Inception网络的前向传播
        with torch.no_grad():
            # 通过Inception的初始层
            x = self.inception.Conv2d_1a_3x3(x)  # 299 x 299 x 3
            x = self.inception.Conv2d_2a_3x3(x)  # 149 x 149 x 32
            x = self.inception.Conv2d_2b_3x3(x)  # 149 x 149 x 64
            x = self.inception.maxpool1(x)  # 73 x 73 x 64
            x = self.inception.Conv2d_3b_1x1(x)  # 73 x 73 x 80
            x = self.inception.Conv2d_4a_3x3(x)  # 71 x 71 x 192
            x = self.inception.maxpool2(x)  # 35 x 35 x 192
            
            # Inception模块
            x = self.inception.Mixed_5b(x)
            x = self.inception.Mixed_5c(x)
            x = self.inception.Mixed_5d(x)
            x = self.inception.Mixed_6a(x)
            x = self.inception.Mixed_6b(x)
            x = self.inception.Mixed_6c(x)
            x = self.inception.Mixed_6d(x)
            x = self.inception.Mixed_6e(x)
            x = self.inception.Mixed_7a(x)
            x = self.inception.Mixed_7b(x)
            x = self.inception.Mixed_7c(x)
            
            # 根据指定的层返回特征
            if self.feature_layer == 'avg_pool':
                # 平均池化，对应TensorFlow中的pool_3:0
                x = self.inception.avgpool(x)
                x = torch.flatten(x, 1)  # [batch_size, 2048]
            elif self.feature_layer == 'fc':
                # 全连接层输出
                x = self.inception.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.inception.dropout(x)
                x = self.inception.fc(x)  # [batch_size, 1000]
            else:
                raise ValueError(f'Unknown layer: {self.feature_layer}')
        
        return x


def preprocess_for_inception(images):
    """
    为Inception网络预处理图像（对标TensorFlow的tf.contrib.gan.eval.preprocess_image）
    
    参数：
        images: 输入张量或numpy数组，形状 [batch_size, height, width, 3]
               像素值范围 [0, 255]（uint8 或 float32）
    
    返回：
        预处理后的张量，形状 [batch_size, 3, 299, 299]
        像素值范围 [-1, 1]
    """
    # 如果输入是numpy数组，转换为张量
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    # 确保张量是float32类型
    if images.dtype != torch.float32:
        images = images.float()
    
    # 处理不同的输入格式
    # 如果输入格式是 [B, H, W, 3]，转换为 [B, 3, H, W]
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2).contiguous()
    
    # 验证通道数
    assert images.shape[1] == 3, f'Expected 3 channels, got {images.shape[1]}'
    
    # 调整图像尺寸到299x299（Inception V3的标准输入尺寸）
    # 对应TensorFlow中tf.contrib.gan.eval.preprocess_image的行为
    if images.shape[2] != 299 or images.shape[3] != 299:
        images = F.interpolate(
            images,
            size=(299, 299),
            mode='bilinear',
            align_corners=False
        )
    
    # 归一化到 [-1, 1] 范围
    # 假设输入像素值在 [0, 255]
    if images.max() > 1.5:
        images = images / 127.5 - 1.0
    
    return images


def get_inception_features(inputs, inception_model, layer_name='avg_pool'):
    """
    使用Inception网络提取特征（对应TensorFlow版本的get_inception_features）
    
    参数：
        inputs: 输入张量，形状 [batch_size, height, width, 3]
               像素值范围 [0, 255]
        inception_model: InceptionV3FeatureExtractor 实例或权重路径
        layer_name: 要提取的层名称
    
    返回：
        特征张量，形状 [batch_size, feature_dim]
    """
    # 预处理输入
    preprocessed = preprocess_for_inception(inputs)
    
    # 确保输入在GPU上（如果模型在GPU上）
    device = next(inception_model.parameters()).device
    preprocessed = preprocessed.to(device)
    
    # 提取特征
    features = inception_model(preprocessed)
    
    return features


def embed_images_in_inception(imgs, inception_path, layer_name='avg_pool', batch_size=32):
    """
    通过Inception网络生成图像的特征嵌入（对应TensorFlow版本的embed_images_in_inception）
    
    参数：
        imgs: 输入的图像数组，形状 [N, H, W, 3]，像素值范围 [0, 255]
        inception_path: Inception网络模型文件的路径（.pth文件）
        layer_name: 要提取特征的网络层名称（默认为'avg_pool'，对应TF的'pool_3:0'）
        batch_size: 每次处理的图像批量大小（默认32）
    
    返回：
        embeddings: 图像的特征嵌入，形状 [N, D]，其中D是特征维度
    """
    # 检查输入是否为numpy数组
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs).float()
    elif not isinstance(imgs, torch.Tensor):
        imgs = torch.tensor(imgs, dtype=torch.float32)
    
    # 确定设备（优先使用GPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化Inception特征提取器
    inception_model = InceptionV3FeatureExtractor(
        inception_path=inception_path,
        layer_name=layer_name
    )
    inception_model = inception_model.to(device)
    inception_model.eval()
    
    # 初始化列表，用于存储所有特征嵌入
    embeddings = []
    
    # 按批次处理图像
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            # 获取当前批次
            batch = imgs[i:i+batch_size].to(device)
            
            # 提取特征
            batch_embeddings = inception_model(batch)
            
            # 转换为numpy数组并存储
            embeddings.append(batch_embeddings.cpu().numpy())
    
    # 将所有批次的特征嵌入拼接成一个数组并返回
    return np.concatenate(embeddings, axis=0)
