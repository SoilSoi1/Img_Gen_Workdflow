#!/bin/bash
# Phase 2: Epoch search for optimal early stopping point
# 4 configs, 1500 epochs each, evaluate FID at every checkpoint
set -e

PROJECT_ROOT="/root/autodl-tmp/Img_Gen_Workdflow"
LDM_DIR="$PROJECT_ROOT/models/diffusion/ldm"
PHASE2_DIR="$PROJECT_ROOT/experiments/ldm/phase2_epoch_search"
SUMMARY="$PHASE2_DIR/fid_vs_epochs.csv"

mkdir -p "$PHASE2_DIR"

# Init summary if not exists
if [ ! -f "$SUMMARY" ]; then
    echo "dataset,base_lr,model_channels,epoch,ckpt_path,fid" > "$SUMMARY"
fi

# Configurations: "dataset base_lr model_channels epochs ckpt_interval"
experiments=(
    "LEAK_PROCESSED 1e-5 192 1500 100"
    "LEAK_PROCESSED 2e-5 192 1500 100"
    "TIGHT_PROCESSED 2e-5 192 1500 100"
    "TIGHT_PROCESSED 1e-5 192 1500 100"
)

for exp in "${experiments[@]}"; do
    read dataset base_lr ch epochs ckpt_int <<< "$exp"

    timestamp=$(date +%Y%m%d)
    save_dir="$PHASE2_DIR/${timestamp}-${dataset,,}_lr${base_lr}_ch${ch}_ep${epochs}"
    final_ckpt="$save_dir/checkpoints/epoch_$(printf '%05d' $epochs).pt"

    # Skip if final checkpoint already exists
    if [ -f "$final_ckpt" ]; then
        echo "[Skip] Already done: $save_dir"
    else
        echo "============================================"
        echo "[Start] $dataset | lr=$base_lr | ch=$ch | epochs=$epochs"
        echo "[Save]  $save_dir"
        echo "============================================"

        mkdir -p "$save_dir"

        # --- Training ---
        cd "$LDM_DIR"
        python train.py \
            --train_root "$PROJECT_ROOT/dataset/$dataset" \
            --base_lr "$base_lr" \
            --batch_size 4 \
            --model_channels "$ch" \
            --epochs "$epochs" \
            --ckpt_interval "$ckpt_int" \
            --save_dir "$save_dir" \
            --device cuda:0 \
            2>&1 | tee "$save_dir/train.log"

        echo "[Train Done] $save_dir"
    fi

    # --- Evaluate all checkpoints ---
    echo "[Eval] Evaluating all checkpoints..."
    real_dir="$PROJECT_ROOT/dataset/$dataset"

    for ckpt in "$save_dir"/checkpoints/epoch_*.pt; do
        [ -f "$ckpt" ] || continue

        # Extract epoch number from filename
        epoch_str=$(basename "$ckpt" | grep -oP 'epoch_\K[0-9]+' | sed 's/^0*//')
        epoch_num=$((10#$epoch_str))

        # Skip if already evaluated
        if grep -q "$ckpt" "$SUMMARY" 2>/dev/null; then
            echo "  [Skip] epoch $epoch_num already evaluated"
            continue
        fi

        gen_dir="$save_dir/generated_epoch_${epoch_str}"
        echo "  [Gen] epoch $epoch_num -> $gen_dir"

        python batch_infer.py \
            --ckpt "$ckpt" \
            --output_dir "$gen_dir" \
            --num_samples 50 \
            --batch_size 4 \
            --seed 42 \
            --device cuda:0 \
            > "$save_dir/gen_epoch_${epoch_str}.log" 2>&1

        echo "  [FID] Computing FID for epoch $epoch_num..."
        fid_score=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT/evaluators')
from _fid import cal_fid
try:
    score = cal_fid('$real_dir', '$gen_dir', device='cuda')
    print(f'{score:.4f}')
except Exception as e:
    print(f'ERROR:{e}')
")

        echo "$dataset,$base_lr,$ch,$epoch_num,$ckpt,$fid_score" >> "$SUMMARY"
        echo "  [Done] epoch $epoch_num FID=$fid_score"
    done

    echo "[Done] $dataset lr=$base_lr all checkpoints evaluated"
    echo ""
done

echo "============================================"
echo "Phase 2 Complete. Summary: $SUMMARY"
echo "============================================"
