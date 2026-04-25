"""
Batch inference: load a checkpoint once, generate N images in batches.
"""
import os
import argparse
import torch
from torchvision.utils import save_image
from infer import LDMInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer = LDMInference(
        ckpt_path=args.ckpt,
        device=args.device,
        ddim_steps=args.ddim_steps
    )

    total = args.num_samples
    bs = args.batch_size
    saved = 0

    for i in range((total + bs - 1) // bs):
        current_bs = min(bs, total - saved)
        print(f"[Batch {i+1}/{(total + bs - 1)//bs}] Generating {current_bs} samples...")
        images = infer.sample(
            num_samples=current_bs,
            eta=0.0,
            seed=args.seed + i  # vary seed per batch for diversity
        )
        infer.save_images(images, args.output_dir, prefix=f'sample_{saved:05d}')
        saved += current_bs

    print(f"\n[Done] Generated {saved} images in {args.output_dir}")


if __name__ == '__main__':
    main()
