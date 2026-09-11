#!/usr/bin/env bash
# LM 5 h: LapA v1 M=256 with the WINDOW-ALIGNED damping (new default: lam_max = 1/Ls, damped
# memories in [Ls, 32*Ls]) -- same shape as l5_v1m256_s0 (ff 388, matched to GDN 3x80), so the
# pair isolates the damping change at M=256, and l5_v1m256_s0 - l5_Amixedw64_s0 isolates M.
# Waits for the smoothing ablation queue to finish.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
cd "$(dirname "$0")"
until grep -q "COPY ABLATE SMOOTH DONE" runs/copy_d128_ablate_smooth.log 2>/dev/null; do sleep 120; done
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 128 --layers 4 --Md 4 --G 8 --freq rope
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8 --samples 0 --only sca2 --log runs/long5h.jsonl --class-eval"
echo "##### $(date +%H:%M) l5_v1m256al_s0 (aligned damping)"
python -u pretrain.py --label l5_v1m256al_s0 --seed 0 --variant cshort_damph_cc $COMMON --Mc 256 --dv 56 --ff 388 --Ls 64 --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25 --save runs/ck_l5_v1m256al_s0
echo "##### M256 ALIGNED DONE"
