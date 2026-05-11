#!/bin/bash
# Phase 3: Model size comparison (model_channels = 128 / 192 / 256)
# All 6 metrics evaluated for each experiment
set -e

PROJECT_ROOT="/root/autodl-tmp/Img_Gen_Workdflow"
LDM_DIR="$PROJECT_ROOT/models/diffusion/ldm"
PHASE3_DIR="$PROJECT_ROOT/experiments/ldm/phase3_model_size"
SUMMARY="$PHASE3_DIR/summary.csv"

mkdir -p "$PHASE3_DIR"

# Init summary if not exists
if [ ! -f "$SUMMARY" ]; then
    echo "dataset,base_lr,model_channels,epochs,save_dir,fid,kid,lpips,brisque,prd_f8,prd_f1_8" > "$SUMMARY"
fi

# Anti-duplicate: check if another phase3 train is running
if pgrep -f "python.*train\.py.*phase3_model_size" > /dev/null; then
    echo "[Error] Another phase3 training is already running. Abort."
    exit 1
fi

# Configurations: "dataset base_lr model_channels epochs"
experiments=(
    "LEAK_PROCESSED 2e-5 128 800"
    "LEAK_PROCESSED 2e-5 192 800"
    "LEAK_PROCESSED 2e-5 256 800"
    "TIGHT_PROCESSED 2e-5 128 1000"
    "TIGHT_PROCESSED 2e-5 192 1000"
    "TIGHT_PROCESSED 2e-5 256 1000"
)

for exp in "${experiments[@]}"; do
    read dataset base_lr ch epochs <<< "$exp"

    timestamp=$(date +%Y%m%d)
    save_dir="$PHASE3_DIR/${timestamp}-${dataset,,}_lr${base_lr}_ch${ch}_ep${epochs}"
    final_ckpt="$save_dir/checkpoints/epoch_$(printf '%05d' $epochs).pt"

    # Skip if final checkpoint already exists
    if [ -f "$final_ckpt" ]; then
        echo "[Skip] Training done: $save_dir"
    else
        echo "============================================"
        echo "[Start] $dataset | lr=$base_lr | ch=$ch | epochs=$epochs"
        echo "[Save]  $save_dir"
        echo "============================================"

        mkdir -p "$save_dir"

        cd "$LDM_DIR"
        python train.py \
            --train_root "$PROJECT_ROOT/dataset/$dataset" \
            --base_lr "$base_lr" \
            --batch_size 4 \
            --model_channels "$ch" \
            --epochs "$epochs" \
            --ckpt_interval "$epochs" \
            --save_dir "$save_dir" \
            --device cuda:0 \
            2>&1 | tee "$save_dir/train.log"

        echo "[Train Done] $save_dir"
    fi

    # --- Full evaluation ---
    best_ckpt="$save_dir/checkpoints/best.pt"
    if [ ! -f "$best_ckpt" ]; then
        echo "[Warning] best.pt not found, using last.pt"
        best_ckpt="$save_dir/checkpoints/last.pt"
    fi

    gen_dir="$save_dir/generated"
    real_dir="$PROJECT_ROOT/dataset/$dataset"

    # Generate samples if not exists
    if [ ! -d "$gen_dir" ] || [ "$(ls -1 "$gen_dir" 2>/dev/null | wc -l)" -lt 50 ]; then
        echo "[Gen] Generating 50 samples..."
        python batch_infer.py \
            --ckpt "$best_ckpt" \
            --output_dir "$gen_dir" \
            --num_samples 50 \
            --batch_size 4 \
            --seed 42 \
            --device cuda:0 \
            > "$save_dir/generate.log" 2>&1
    else
        echo "[Skip] Samples already generated"
    fi

    echo "[Eval] Running all 6 metrics..."

    eval_result=$(python3 -c "
import sys, subprocess, os
sys.path.insert(0, '$PROJECT_ROOT/evaluators')
from _fid import cal_fid
from _kid import cal_kid
from lpips_pairwise import cal_lpips_pairwise
from brisque_official import cal_brisque_official

real = '$real_dir'
fake = '$gen_dir'

results = {}
try:
    results['fid'] = cal_fid(real, fake, device='cuda')
except Exception as e:
    results['fid'] = f'ERR:{e}'

try:
    results['kid'] = cal_kid(real, fake, device='cuda')
except Exception as e:
    results['kid'] = f'ERR:{e}'

try:
    results['lpips'] = cal_lpips_pairwise(fake, device='cuda', net='alex', sample_pairs=500)
except Exception as e:
    results['lpips'] = f'ERR:{e}'

try:
    results['brisque'] = cal_brisque_official(fake)
except Exception as e:
    results['brisque'] = f'ERR:{e}'

# PRD
prd_script = os.path.join('$PROJECT_ROOT', 'evaluators/prd/prd_from_image_folders.py')
prd_out = subprocess.run([
    'python', prd_script,
    '--reference_dir', real,
    '--eval_dirs', fake,
    '--eval_labels', 'gen',
    '--inception_path', os.path.join('$PROJECT_ROOT', 'evaluators/prd/inception_v3.pth'),
    '--device', 'cuda',
    '--no_enforce_balance',
    '--silent'
], capture_output=True, text=True, timeout=300)
prd_line = [l for l in prd_out.stdout.split('\n') if l.strip() and l[0].isdigit()]
if prd_line:
    parts = prd_line[0].split()
    results['prd_f8'] = parts[0]
    results['prd_f1_8'] = parts[1]
else:
    results['prd_f8'] = 'ERR'
    results['prd_f1_8'] = 'ERR'

print(f\"{results['fid']},{results['kid']},{results['lpips']},{results['brisque']},{results['prd_f8']},{results['prd_f1_8']}\")
" 2>&1)

    eval_csv=$(echo "$eval_result" | tail -1)
    echo "[Eval Result] $eval_csv"
    echo "$dataset,$base_lr,$ch,$epochs,$save_dir,$eval_csv" >> "$SUMMARY"

    echo "[Done] $dataset lr=$base_lr ch=$ch"
    echo ""
done

echo "============================================"
echo "Phase 3 Complete. Summary: $SUMMARY"
echo "============================================"
