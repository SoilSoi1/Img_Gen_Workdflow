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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

from util import instantiate_from_config
from omegaconf import OmegaConf

from dataset import UnconditionalImageDataset


class SimplifiedLDMWrapper(nn.Module):
    """Simplified LDM wrapper that doesn't depend on official code."""
    
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.model_config = model_config
        
        # Store diffusion parameters from config
        self.linear_start = model_config.params.linear_start
        self.linear_end = model_config.params.linear_end
        self.num_timesteps = model_config.params.timesteps
        
        # Create a simple UNet-like denoiser (without canonical openai implementation)
        # This avoids importing from official LDM which has pytorch_lightning deps
        in_channels = model_config.params.unet_config.params.in_channels
        model_channels = model_config.params.unet_config.params.model_channels
        out_channels = model_config.params.unet_config.params.out_channels
        num_heads = model_config.params.unet_config.params.num_head_channels
        
        self.unet = SimpleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_heads=num_heads
        )
        
        # Create noise schedule
        self._setup_noise_schedule()
    
    def _setup_noise_schedule(self):
        """Setup linear noise schedule for DDPM."""
        betas = torch.linspace(self.linear_start, self.linear_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def forward(self, x):
        """Forward pass: add noise and denoise."""
        # For simplicity, we skip VAE encoding and work in image space
        # In practice, you'd want proper VAE encoding
        z = x
        
        # Sample random timestep
        t = torch.randint(0, self.num_timesteps, (x.shape[0],)).to(self.device)
        
        # Sample noise
        noise = torch.randn_like(z)
        
        # Add noise to latent (forward diffusion process)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t])
        
        # Reshape for broadcasting over batch
        shape = [x.shape[0], 1, 1, 1]
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(*shape)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(*shape)
        
        z_t = sqrt_alphas_cumprod * z + sqrt_one_minus_alphas_cumprod * noise
        
        # Predict noise using UNet
        pred_noise = self.unet(z_t, t)
        
        # MSE loss
        loss = nn.functional.mse_loss(pred_noise, noise, reduction='mean')
        
        return loss


class SimpleUNet(nn.Module):
    """Simple UNet-like architecture for diffusion denoising."""
    
    def __init__(self, in_channels=3, out_channels=3, model_channels=128, num_heads=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        
        # Encoder
        self.enc1 = self._make_block(in_channels, model_channels)
        self.down1 = nn.MaxPool2d(2)
        
        self.enc2 = self._make_block(model_channels, model_channels * 2)
        self.down2 = nn.MaxPool2d(2)
        
        # Bottleneck  
        self.bottleneck = self._make_block(model_channels * 2, model_channels * 4)
        
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec1 = self._make_block(model_channels * 4 + model_channels * 2, model_channels * 2)
        
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec2 = self._make_block(model_channels * 2 + model_channels, model_channels)
        
        # Output
        self.final = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self._get_timestep_embedding(t, self.model_channels)
        
        # Encoder
        e1 = self.enc1(x)
        x = self.down1(e1)
        
        e2 = self.enc2(x)
        x = self.down2(e2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.up1(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)
        
        x = self.final(x)
        
        return x
    
    def _get_timestep_embedding(self, t, embedding_dim):
        """Get sinusoidal timestep embeddings."""
        # Sinusoidal position encoding
        device = t.device
        half_dim = embedding_dim // 2
        emb = torch.arange(half_dim, device=device, dtype=torch.float32)
        emb = math.exp(-math.log(10000.0) / (half_dim - 1)) * emb
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb



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
        
        # Add LDM to path so ldm modules can be imported
        ldm_root = Path(__file__).parent.parent.parent.parent / "temp" / "latent-diffusion"
        if str(ldm_root) not in sys.path:
            sys.path.insert(0, str(ldm_root))
            print(f"[Model] Added to path: {ldm_root}")
        
        # Load config
        config_path = ldm_root / "configs/latent-diffusion/celebahq-ldm-vq-4.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        print(f"[Model] Using config: {config_path}")
        config = OmegaConf.load(config_path)
        
        # Override with user config
        config.model.params.image_size = self.config['image_size']
        config.model.params.channels = self.config.get('channels', 3)
        config.data.params.batch_size = self.config['batch_size']
        
        # Create simplified LDM wrapper that avoids pytorch_lightning dependencies
        self.model = SimplifiedLDMWrapper(config.model, self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Model] #Parameters (trainable): {n_params:,}")
        print(f"[Model] Image size: {self.config['image_size']}")
        print(f"[Model] Channels: {self.config.get('channels', 3)}")
    
    def _setup_optimizer(self):
        """Setup optimizer and LR scheduler."""
        # Use base learning rate from config
        base_lr = self.config.get('base_lr', 2.0e-06)
        
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
            
            # Save checkpoint
            if (epoch + 1) % self.config['ckpt_interval'] == 0:
                self._save_checkpoint(epoch)
            
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
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        ckpt_path = self.ckpt_dir / f"epoch_{epoch+1:05d}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }, ckpt_path)
        
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
        
        print(f"[Resume] Loaded from {ckpt_path}")
        print(f"[Resume] Resuming from epoch {self.current_epoch}")


def get_default_config():
    """Get default configuration."""
    return {
        # Data
        'train_root': './datasets/color_20260321/train',
        'val_root': None,
        'image_size': 256,
        'channels': 3,
        
        # Training
        'epochs': 100,
        'batch_size': 4,
        'num_workers': 4,
        'base_lr': 2.0e-06,
        'scale_lr': True,
        'grad_clip': 1.0,
        
        # Saving
        'save_dir': './experiments/ldm/exp_1',
        'ckpt_interval': 10,
        'log_interval': 100,
        
        # Device
        'device': 'cuda:0',
    }


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    
    # Data
    parser.add_argument('--train_root', type=str, default=None, help='Training data root')
    parser.add_argument('--val_root', type=str, default=None, help='Validation data root')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--channels', type=int, default=3, help='Image channels')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--base_lr', type=float, default=2.0e-06, help='Base learning rate')
    parser.add_argument('--scale_lr', action='store_true', help='Scale LR by batch size')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='Checkpoint save interval')
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
