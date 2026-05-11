#!/usr/bin/env python3
"""
Full evaluation (6 metrics) for best-epoch checkpoints from Phase 2.
FID already computed; this computes KID, LPIPS, BRISQUE, PRD.
"""
import sys
import os
import subprocess

sys.path.insert(0, '/root/autodl-tmp/Img_Gen_Workdflow/evaluators')
from _kid import cal_kid
from lpips_pairwise import cal_lpips_pairwise
from brisque_official import cal_brisque_official

PROJECT_ROOT = "/root/autodl-tmp/Img_Gen_Workdflow"

experiments = [
    ("LEAK_PROCESSED", "1e-5", "20260508-leak_processed_lr1e-5_ch192_ep1500", 1400, "dataset/LEAK_PROCESSED"),
    ("LEAK_PROCESSED", "2e-5", "20260509-leak_processed_lr2e-5_ch192_ep1500", 800, "dataset/LEAK_PROCESSED"),
    ("TIGHT_PROCESSED", "1e-5", "20260510-tight_processed_lr1e-5_ch192_ep1500", 1300, "dataset/TIGHT_PROCESSED"),
    ("TIGHT_PROCESSED", "2e-5", "20260509-tight_processed_lr2e-5_ch192_ep1500", 1000, "dataset/TIGHT_PROCESSED"),
]

summary_path = os.path.join(PROJECT_ROOT, "experiments/ldm/phase2_epoch_search/full_eval_summary.csv")
with open(summary_path, 'w') as f:
    f.write("dataset,base_lr,best_epoch,fid,kid,lpips,brisque,prd_f8,prd_f1_8\n")

for dataset, base_lr, exp_dir, best_epoch, real_rel in experiments:
    gen_dir = os.path.join(PROJECT_ROOT, "experiments/ldm/phase2_epoch_search", exp_dir, f"generated_epoch_{best_epoch:04d}")
    real_dir = os.path.join(PROJECT_ROOT, real_rel)
    
    print(f"\n{'='*60}")
    print(f"[Eval] {dataset} lr={base_lr} epoch={best_epoch}")
    print(f"[Gen]  {gen_dir}")
    print(f"{'='*60}")
    
    # FID (already computed, read from csv)
    fid = "N/A"
    
    # KID
    print("[KID]...")
    try:
        kid = cal_kid(real_dir, gen_dir, device='cuda')
    except Exception as e:
        kid = f"ERR:{e}"
    print(f"  KID={kid}")
    
    # Intra-LPIPS
    print("[LPIPS]...")
    try:
        lpips = cal_lpips_pairwise(gen_dir, device='cuda', net='alex', sample_pairs=500)
    except Exception as e:
        lpips = f"ERR:{e}"
    print(f"  LPIPS={lpips}")
    
    # BRISQUE
    print("[BRISQUE]...")
    try:
        brisque = cal_brisque_official(gen_dir)
    except Exception as e:
        brisque = f"ERR:{e}"
    print(f"  BRISQUE={brisque}")
    
    # PRD
    print("[PRD]...")
    try:
        prd_script = os.path.join(PROJECT_ROOT, "evaluators/prd/prd_from_image_folders.py")
        prd_out = subprocess.run([
            "python", prd_script,
            "--reference_dir", real_dir,
            "--eval_dirs", gen_dir,
            "--eval_labels", "gen",
            "--inception_path", os.path.join(PROJECT_ROOT, "evaluators/prd/inception_v3.pth"),
            "--device", "cuda",
            "--no_enforce_balance",
            "--silent"
        ], capture_output=True, text=True, timeout=300)
        prd_line = [l for l in prd_out.stdout.split('\n') if l.strip() and l[0].isdigit()]
        if prd_line:
            parts = prd_line[0].split()
            prd_f8 = parts[0]
            prd_f1_8 = parts[1]
        else:
            prd_f8 = "ERR"
            prd_f1_8 = "ERR"
    except Exception as e:
        prd_f8 = f"ERR:{e}"
        prd_f1_8 = f"ERR:{e}"
    print(f"  PRD F_8={prd_f8} F_1/8={prd_f1_8}")
    
    with open(summary_path, 'a') as f:
        f.write(f"{dataset},{base_lr},{best_epoch},{fid},{kid},{lpips},{brisque},{prd_f8},{prd_f1_8}\n")
    
    print(f"[Done] {dataset} lr={base_lr}")

print(f"\n{'='*60}")
print(f"All evaluations complete. Summary: {summary_path}")
print(f"{'='*60}")
