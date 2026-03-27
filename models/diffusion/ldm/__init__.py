"""
Latent Diffusion Model (LDM) training and inference scripts.

Core modules:
- dataset.py: UnconditionalImageDataset for flexible data loading
- train.py: LDMTrainer for training unconditional LDMs
- infer.py: LDMInference for image generation and sampling
"""

from .dataset import UnconditionalImageDataset

__version__ = "0.1.0"
__all__ = ["UnconditionalImageDataset"]
