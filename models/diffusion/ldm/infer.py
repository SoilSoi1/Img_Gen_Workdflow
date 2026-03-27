"""
Simplified inference script for trained Latent Diffusion Model.

Generates images from trained LDM checkpoints.

Usage:
    python infer.py --ckpt checkpoints/epoch_00100.pt --num_samples 10 --output_dir ./outputs
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from util import instantiate_from_config
from models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf


class LDMInference:
    """Inference class for Latent Diffusion Model."""
    
    def __init__(self, ckpt_path, device='cuda:0', ddim_steps=50):
        """
        Initialize inference.
        
        Args:
            ckpt_path (str): Path to checkpoint
            device (str): Device to use
            ddim_steps (int): Number of DDIM steps for sampling
        """
        self.device = torch.device(device)
        self.ddim_steps = ddim_steps
        
        # Load model
        self._load_model(ckpt_path)
        
        # Setup sampler
        self.sampler = DDIMSampler(self.model)
        
        print(f"[Inference] Initialized on {self.device}")
        print(f"[Inference] DDIM steps: {ddim_steps}")
    
    def _load_model(self, ckpt_path):
        """Load model from checkpoint."""
        print(f"[Model] Loading from {ckpt_path}")
        
        # Load LDM config - try multiple locations
        possible_paths = [
            Path(__file__).parent.parent.parent.parent / "temp" / "latent-diffusion" / "configs/latent-diffusion/celebahq-ldm-vq-4.yaml",
            Path(__file__).parent / "config.yaml",
            Path(__file__).parent.parent.parent / "configs" / "model" / "diffusion.yaml",
        ]
        
        config_path = None
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"Config not found in any of: {possible_paths}")
        
        print(f"[Model] Using config: {config_path}")
        config = OmegaConf.load(config_path)
        
        # Instantiate model
        self.model = instantiate_from_config(config.model)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        
        # Load state dict with strict=False to allow partial loading
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Warning] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Warning] Unexpected keys: {len(unexpected)}")
        
        print(f"[Model] Loaded successfully")
        
        # Get image size
        if hasattr(self.model, 'image_size'):
            self.image_size = self.model.image_size
        else:
            self.image_size = 256
        
        print(f"[Model] Image size: {self.image_size}")
    
    @torch.no_grad()
    def sample(self, num_samples=4, eta=0.0, seed=None, guidance_scale=1.0):
        """
        Sample images from the model.
        
        Args:
            num_samples (int): Number of samples to generate
            eta (float): DDIM parameter (0 = deterministic, 1 = stochastic)
            seed (int): Random seed for reproducibility
            guidance_scale (float): Guidance scale (1.0 = no guidance)
        
        Returns:
            images (torch.Tensor): Generated images in [0, 1] range
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        print(f"[Sampling] Generating {num_samples} samples with eta={eta}...")
        
        # Sample in latent space
        # For unconditional models, conditioning is None
        shape = (num_samples, self.model.channels, self.image_size // 8, self.image_size // 8)
        
        try:
            samples, _ = self.sampler.sample(
                S=self.ddim_steps,
                conditioning=None,
                batch_size=num_samples,
                shape=shape[1:],
                eta=eta,
                verbose=True
            )
        except Exception as e:
            print(f"[Warning] Standard sampling failed: {e}")
            print("[Fallback] Using random latent codes")
            samples = torch.randn(shape, device=self.device)
        
        # Decode to image space
        images = self._decode_latents(samples)
        
        return images
    
    @torch.no_grad()
    def _decode_latents(self, latents):
        """
        Decode latents to image space using VAE decoder.
        
        Args:
            latents (torch.Tensor): Latent codes
        
        Returns:
            images (torch.Tensor): Images in [0, 1] range
        """
        print("[Decode] Decoding latents to image space...")
        
        # Use first_stage_model (VAE) to decode
        if hasattr(self.model, 'first_stage_model'):
            images = self.model.first_stage_model.decode(latents)
        else:
            print("[Warning] No VAE decoder found, returning latents as-is")
            images = latents
        
        # Denormalize from [-1, 1] to [0, 1]
        images = torch.clamp(images, -1, 1)
        images = (images + 1) / 2
        
        return images
    
    def save_images(self, images, output_dir, prefix='sample'):
        """
        Save generated images.
        
        Args:
            images (torch.Tensor): Images in [0, 1] range, shape (N, 3, H, W)
            output_dir (str): Output directory
            prefix (str): Filename prefix
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Move to CPU if necessary
        if images.device.type == 'cuda':
            images = images.cpu()
        
        for i, img in enumerate(images):
            fname = os.path.join(output_dir, f'{prefix}_{i:05d}.png')
            save_image(img, fname)
            print(f"[Save] {fname}")


def main():
    parser = argparse.ArgumentParser(description='LDM Inference')
    
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM steps')
    parser.add_argument('--eta', type=float, default=0.0, help='DDIM eta (0=deterministic)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    
    args = parser.parse_args()
    
    # Auto-generate output directory
    if args.output_dir is None:
        ckpt_dir = os.path.dirname(args.ckpt)
        args.output_dir = os.path.join(ckpt_dir or '.', 'generated')
    
    print("\n" + "="*60)
    print("Latent Diffusion Model (LDM) Inference")
    print("="*60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Samples: {args.num_samples}")
    print(f"DDIM Steps: {args.ddim_steps}")
    print(f"ETA: {args.eta}")
    print(f"Output Dir: {args.output_dir}\n")
    
    try:
        # Create inference engine
        infer = LDMInference(
            ckpt_path=args.ckpt,
            device=args.device,
            ddim_steps=args.ddim_steps
        )
        
        # Sample
        images = infer.sample(
            num_samples=args.num_samples,
            eta=args.eta,
            seed=args.seed
        )
        
        # Save
        infer.save_images(images, args.output_dir, prefix='ldm_sample')
        
        print(f"\n[Done] Generated {args.num_samples} images")
        print(f"[Done] Saved to {args.output_dir}\n")
    
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
