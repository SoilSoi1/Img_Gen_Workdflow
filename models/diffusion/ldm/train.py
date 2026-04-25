"""
Latent Diffusion Model (LDM) training script for unconditional image generation.

This script trains an LDM on private image datasets using the official 
ldm.models.diffusion.ddpm.LatentDiffusion architecture.

Usage:
    python train.py --data_root /path/to/images --device cuda:0
    python train.py --data_root /path/to/images --epochs 100 --batch_size 16
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

from util import instantiate_from_config
from omegaconf import OmegaConf

from dataset import UnconditionalImageDataset
from model import SimplifiedLDMWrapper


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return deepcopy(self.shadow)
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict


class LDMTrainer:
    """Trainer for unconditional Latent Diffusion Models."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Setup directories
        self.ckpt_dir = Path(config['save_dir']) / 'checkpoints'
        self.log_dir = Path(config['save_dir']) / 'logs'
        self.sample_dir = Path(config['save_dir']) / 'samples'
        
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logger
        self.writer = SummaryWriter(str(self.log_dir))
        
        # Data
        self._setup_data()
        
        # Model
        self._setup_model()
        
        # Optimizer
        self._setup_optimizer()
        
        # EMA
        self.ema = EMA(self.model, decay=0.9999)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        print(f"[Trainer] Initialized on device: {self.device}")
        print(f"[Trainer] Checkpoints: {self.ckpt_dir}")
        print(f"[Trainer] Logs: {self.log_dir}")
    
    def _setup_data(self):
        """Setup train/val dataloaders."""
        print("[Data] Loading datasets...")
        
        # Train dataset
        train_dataset = UnconditionalImageDataset(
            data_dir=self.config['train_root'],
            image_size=self.config['image_size']
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        # Validation dataset (optional)
        self.val_loader = None
        if self.config.get('val_root') and os.path.exists(self.config['val_root']):
            val_dataset = UnconditionalImageDataset(
                data_dir=self.config['val_root'],
                image_size=self.config['image_size']
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['num_workers'],
                pin_memory=True
            )
        
        print(f"[Data] Train batches per epoch: {len(self.train_loader)}")
        if self.val_loader:
            print(f"[Data] Val batches per epoch: {len(self.val_loader)}")
    
    def _setup_model(self):
        """Setup LDM model and autoencoder."""
        print("[Model] Loading Latent Diffusion Model...")
        
        # Load config from local configs directory
        ldm_root = Path(__file__).parent
        
        # Load config
        config_path = ldm_root / "configs" / "latent-diffusion" / "celebahq-ldm-vq-4.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        print(f"[Model] Using config: {config_path}")
        config = OmegaConf.load(config_path)
        
        # Override with user config
        config.model.params.image_size = self.config['image_size']
        config.model.params.channels = self.config.get('channels', 3)
        # UNet now operates on latent space (4 channels instead of 3)
        config.model.params.unet_config.params.in_channels = 4
        config.model.params.unet_config.params.out_channels = 4
        
        # Create full LDM wrapper with VAE + UNet
        self.model = SimplifiedLDMWrapper(config.model, self.device, image_size=self.config['image_size'])
        self.model = self.model.to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] #Parameters (trainable): {n_params:,}")
        print(f"[Model] Image size: {self.config['image_size']}")
        print(f"[Model] Channels: {self.config.get('channels', 3)}")
    
    def _setup_optimizer(self):
        """Setup optimizer and LR scheduler."""
        # Use base learning rate from config
        base_lr = self.config.get('base_lr', 1.0e-05)
        
        # Scale learning rate by batch size (common practice)
        if self.config.get('scale_lr', True):
            lr = base_lr * self.config['batch_size']
        else:
            lr = base_lr
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )
        
        print(f"[Optimizer] AdamW with lr={lr:.2e}")
    
    def train(self):
        """Main training loop."""
        print(f"\n[Training] Starting from epoch {self.current_epoch}\n")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self._train_epoch()
            
            # Validation epoch
            val_loss = None
            if self.val_loader:
                val_loss = self._val_epoch()
            
            # Logging
            self.writer.add_scalar('train/loss', train_loss, epoch)
            if val_loss is not None:
                self.writer.add_scalar('val/loss', val_loss, epoch)
            
            # Learning rate logging
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', lr, epoch)
            
            # Print progress
            msg = f"[Epoch {epoch+1}/{self.config['epochs']}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            msg += f" lr={lr:.2e}"
            print(msg)
            
            # Save checkpoint at interval
            if (epoch + 1) % self.config['ckpt_interval'] == 0:
                self._save_checkpoint(epoch)
            
            # Save best model (using EMA weights)
            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch, filename='best.pt', use_ema=True)
            
            # Always save last.pt for resume
            self._save_checkpoint(epoch, filename='last.pt')
            
            # LR scheduler step
            self.scheduler.step()
        
        print("[Training] Finished!")
        self.writer.close()
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(self.train_loader):
            x = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model(x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update EMA
            self.ema.update(self.model)
            
            total_loss += loss.item()
            self.global_step += 1
            
            # Mini-batch logging
            if (batch_idx + 1) % self.config.get('log_interval', 100) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  [{batch_idx+1}/{len(self.train_loader)}] loss={avg_loss:.4f}")
        
        return total_loss / len(self.train_loader)
    
    def _val_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch.to(self.device)
                loss = self.model(x)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def _save_checkpoint(self, epoch, filename=None, use_ema=False):
        """Save model checkpoint."""
        if filename is None:
            ckpt_path = self.ckpt_dir / f"epoch_{epoch+1:05d}.pt"
        else:
            ckpt_path = self.ckpt_dir / filename
        
        # Optionally use EMA weights for inference-quality checkpoints
        if use_ema:
            self.ema.apply_shadow(self.model)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'image_size': self.config['image_size'],
            'ema_shadow': self.ema.state_dict(),
        }
        
        if use_ema:
            self.ema.restore(self.model)
        
        torch.save(state, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
    
    def resume_from_checkpoint(self, ckpt_path):
        """Resume training from checkpoint."""
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint not found: {ckpt_path}")
            return
        
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        self.current_epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        
        if 'ema_shadow' in ckpt:
            self.ema.load_state_dict(ckpt['ema_shadow'])
            print("[Resume] EMA state loaded")
        
        print(f"[Resume] Loaded from {ckpt_path}")
        print(f"[Resume] Resuming from epoch {self.current_epoch}")


def get_default_config():
    """Get default configuration."""
    return {
        # Data
        'train_root': './datasets/color_20260321/train',
        'val_root': None,
        'image_size': 512,
        'channels': 3,
        
        # Training
        'epochs': 100,
        'batch_size': 4,
        'num_workers': 4,
        'base_lr': 1.0e-05,
        'scale_lr': True,
        'grad_clip': 1.0,
        
        # Saving
        'save_dir': './experiments/ldm/exp_1',
        'ckpt_interval': 100,
        'log_interval': 100,
        
        # Device
        'device': 'cuda:0',
    }


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    
    # Data
    parser.add_argument('--train_root', type=str, default=None, help='Training data root')
    parser.add_argument('--val_root', type=str, default=None, help='Validation data root')
    parser.add_argument('--image_size', type=int, default=512, help='Image size')
    parser.add_argument('--channels', type=int, default=3, help='Image channels')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--base_lr', type=float, default=1.0e-05, help='Base learning rate')
    parser.add_argument('--scale_lr', action='store_true', help='Scale LR by batch size')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument('--ckpt_interval', type=int, default=100, help='Checkpoint save interval')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    
    # Build config
    config = get_default_config()
    
    # Override with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # Auto-generate save_dir if not provided
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        config['save_dir'] = f"./experiments/ldm/{timestamp}"
    
    # Validate required arguments
    if config['train_root'] is None:
        config['train_root'] = './datasets/color_20260321/train'
    
    if not os.path.exists(config['train_root']):
        raise ValueError(f"Train root not found: {config['train_root']}")
    
    print("\n" + "="*60)
    print("Latent Diffusion Model (LDM) Training")
    print("="*60)
    print("\n[Config]")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create trainer
    trainer = LDMTrainer(config)
    
    # Resume if checkpoint provided
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
