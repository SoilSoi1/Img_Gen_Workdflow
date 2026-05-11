#!/bin/bash
# Phase 1: LR search for LDM training
# All evaluators: FID, KID, Intra-LPIPS, BRISQUE
set -e

PROJECT_ROOT="/root/autodl-tmp/Img_Gen_Workdflow"
LDM_DIR="$PROJECT_ROOT/models/diffusion/ldm"
PHASE1_DIR="$PROJECT_ROOT/experiments/ldm/phase1_lr_search"
SUMMARY="$PHASE1_DIR/summary.csv"

mkdir -p "$PHASE1_DIR"

# Init summary if not exists
if [ ! -f "$SUMMARY" ]; then
    echo "dataset,base_lr,actual_lr,model_channels,epochs,save_dir,fid,kid,lpips,brisque,prd_f8,prd_f1_8" > "$SUMMARY"
fi

# Configurations: "dataset base_lr model_channels epochs"
experiments=(
    "LEAK_PROCESSED 5e-6 192 500"
    "LEAK_PROCESSED 1e-5 192 500"
    "LEAK_PROCESSED 2e-5 192 500"
    "TIGHT_PROCESSED 5e-6 192 500"
    "TIGHT_PROCESSED 1e-5 192 500"
    "TIGHT_PROCESSED 2e-5 192 500"
)

for exp in "${experiments[@]}"; do
    read dataset base_lr ch epochs <<< "$exp"

    timestamp=$(date +%Y%m%d)
    save_dir="$PHASE1_DIR/${timestamp}-${dataset,,}_lr${base_lr}_ch${ch}"
    ckpt_path="$save_dir/checkpoints/epoch_$(printf '%05d' $epochs).pt"

    # Skip if already completed
    if [ -f "$ckpt_path" ]; then
        echo "[Skip] Already done: $save_dir"
        continue
    fi

    echo "============================================"
    echo "[Start] $dataset | lr=$base_lr | ch=$ch | epochs=$epochs"
    echo "[Save]  $save_dir"
    echo "============================================"

    # Create save_dir early so tee can write log
    mkdir -p "$save_dir"

    # --- Training ---
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

    # --- Generate samples ---
    best_ckpt="$save_dir/checkpoints/best.pt"
    if [ ! -f "$best_ckpt" ]; then
        echo "[Warning] best.pt not found, using last.pt"
        best_ckpt="$save_dir/checkpoints/last.pt"
    fi

    gen_dir="$save_dir/generated"
    python batch_infer.py \
        --ckpt "$best_ckpt" \
        --output_dir "$gen_dir" \
        --num_samples 50 \
        --batch_size 4 \
        --seed 42 \
        --device cuda:0 \
        2>&1 | tee "$save_dir/generate.log"

    echo "[Gen Done] 50 samples -> $gen_dir"

    # --- Evaluation: all metrics ---
    echo "[Eval] Running all evaluators..."
    real_dir="$PROJECT_ROOT/dataset/$dataset"

    # --- Eval: FID/KID/LPIPS/BRISQUE ---
    eval_result=$(python3 -c "
import sys
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

print(f\"{results['fid']},{results['kid']},{results['lpips']},{results['brisque']}\")
" 2>&1)

    # Extract last line (the CSV output)
    eval_csv=$(echo "$eval_result" | tail -1)
    echo "[Eval Result] FID,KID,LPIPS,BRISQUE = $eval_csv"

    # --- Eval: PRD ---
    echo "[Eval] Running PRD..."
    prd_out=$(python "$PROJECT_ROOT/evaluators/prd/prd_from_image_folders.py" \
        --reference_dir "$real_dir" \
        --eval_dirs "$gen_dir" \
        --eval_labels gen \
        --inception_path "$PROJECT_ROOT/evaluators/prd/inception_v3.pth" \
        --device cuda \
        --no_enforce_balance \
        --silent 2>&1 | grep -E '^[0-9]+\.[0-9]+')

    prd_f8=$(echo "$prd_out" | awk '{print $1}')
    prd_f1_8=$(echo "$prd_out" | awk '{print $2}')
    echo "[PRD Result] F_8=$prd_f8  F_1/8=$prd_f1_8"

    # Compute actual lr for summary
    actual_lr=$(python3 -c "print(f'{float('$base_lr') * 4:.2e}')")
    echo "$dataset,$base_lr,$actual_lr,$ch,$epochs,$save_dir,$eval_csv,$prd_f8,$prd_f1_8" >> "$SUMMARY"

    echo "[Done] $dataset lr=$base_lr"
    echo ""
done

echo "============================================"
echo "Phase 1 Complete. Summary: $SUMMARY"
echo "============================================"
