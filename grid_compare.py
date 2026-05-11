"""
Grid comparison visualization for generated images across different checkpoints.

Layout (paper-style, transposed):
    - Each ROW corresponds to one checkpoint/folder
    - Each COLUMN corresponds to one sample
    - Row labels are simplified: epoch_00500 -> epoch 500

Usage:
    python grid_compare.py \
        --folders \
            experiments/ldm/leak_final/samples/epoch_00500 \
            experiments/ldm/leak_final/samples/epoch_01000 \
            experiments/ldm/leak_final/samples/epoch_01500 \
            experiments/ldm/leak_final/samples/best \
        --n 8 --random --seed 42 \
        --output leak_comparison.png

Font:
    Place TimesNewRoman.ttf in the same directory as this script for Times New Roman.
    Otherwise falls back to DejaVu Serif.
"""
import os
import argparse
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_images(folder):
    """Find all image files in a folder (non-recursive)."""
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f.lower())[1] in exts]
    files.sort()
    return [os.path.join(folder, f) for f in files]


def simplify_label(name):
    """Simplify folder name for display."""
    # epoch_00500 -> epoch 500
    m = re.match(r'epoch_(\d+)', name)
    if m:
        return f"epoch {int(m.group(1))}"
    # warmup_8000epoch -> warmup 8000
    m = re.match(r'warmup_(\d+)epoch', name)
    if m:
        return f"warmup {int(m.group(1))}"
    # ddpm_leak_ckpt_1000_epoch -> ddpm 1000
    m = re.match(r'ddpm_.*?(\d+)_epoch', name)
    if m:
        return f"ddpm {int(m.group(1))}"
    return name


def load_font(size):
    """Load font: prefer local TimesNewRoman.ttf, then system serif, then default."""
    script_dir = Path(__file__).parent
    candidates = [
        str(script_dir / "TimesNewRoman.ttf"),
        str(script_dir / "times.ttf"),
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/System/Library/Fonts/Times.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_grid(folders, n, random_select=False, seed=None, thumb_size=256,
              label_width=None, margin=2, bg_color=(255, 255, 255),
              text_color=(0, 0, 0)):
    """
    Create a transposed grid image.

    Args:
        folders: list of folder paths (each becomes a ROW)
        n: number of columns (samples per row)
        random_select: whether to randomly pick n images per folder
        seed: random seed for reproducibility
        thumb_size: resize each image to this square size
        label_width: width of row label area on the left
        margin: gap between cells
        bg_color: background color
        text_color: text color

    Returns:
        PIL Image
    """
    if seed is not None:
        random.seed(seed)

    rows = len(folders)
    if rows == 0:
        raise ValueError("At least one folder must be provided")

    # Collect images per row
    row_images = []
    row_labels = []
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        imgs = find_images(str(folder))
        if len(imgs) == 0:
            raise ValueError(f"No images found in: {folder}")
        if len(imgs) < n:
            raise ValueError(
                f"Folder '{folder}' has {len(imgs)} images, but n={n} requested"
            )

        selected = random.sample(imgs, n) if random_select else imgs[:n]
        row_images.append(selected)
        row_labels.append(simplify_label(folder.name))

    # Auto-compute label width based on text + padding
    font_label = load_font(22)
    temp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = 0
    for lbl in row_labels:
        bbox = temp_draw.textbbox((0, 0), lbl, font=font_label)
        max_text_w = max(max_text_w, bbox[2] - bbox[0])
    if label_width is None:
        label_width = max_text_w + margin * 3  # padding on both sides

    cols = n
    cell_w = thumb_size
    cell_h = thumb_size

    grid_w = label_width + cols * cell_w + (cols + 1) * margin
    grid_h = rows * cell_h + (rows + 1) * margin

    canvas = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for r in range(rows):
        y = margin + r * (cell_h + margin)

        # Row label (left side, left-aligned)
        bbox = draw.textbbox((0, 0), row_labels[r], font=font_label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = margin
        ty = y + (cell_h - th) // 2
        draw.text((tx, ty), row_labels[r], fill=text_color, font=font_label)

        # Images in this row
        for c in range(cols):
            img_path = row_images[r][c]
            img = Image.open(img_path).convert("RGB")
            img = img.resize((cell_w, cell_h), Image.LANCZOS)
            x = label_width + margin + c * (cell_w + margin)
            canvas.paste(img, (x, y))

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Grid comparison of images across checkpoints (paper-style)"
    )
    parser.add_argument(
        "--folders", nargs="+", required=True,
        help="List of folders containing generated images (each = one ROW)"
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="Number of columns (samples per row) to display"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Randomly select n images from each folder"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible selection"
    )
    parser.add_argument(
        "--thumb_size", type=int, default=256,
        help="Resize images to this size (square)"
    )
    parser.add_argument(
        "--label_width", type=int, default=None,
        help="Width of row label area on the left (auto-computed if omitted)"
    )
    parser.add_argument(
        "--margin", type=int, default=2,
        help="Gap between cells in pixels"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output PNG file path"
    )
    args = parser.parse_args()

    print(f"[Grid] Folders (rows): {args.folders}")
    print(f"[Grid] Columns (n): {args.n}")
    print(f"[Grid] Random: {args.random}")
    print(f"[Grid] Thumb size: {args.thumb_size}")

    grid = make_grid(
        folders=args.folders,
        n=args.n,
        random_select=args.random,
        seed=args.seed,
        thumb_size=args.thumb_size,
        label_width=args.label_width,
        margin=args.margin,
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    grid.save(args.output, quality=95)
    print(f"[Grid] Saved to {args.output}  ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
