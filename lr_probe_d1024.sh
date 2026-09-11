#!/usr/bin/env bash
# 7 min per lr on the LapA arm at the run shape, before committing 15 h to it.
# The d=128 campaign ran constant lr 1e-3 with no warmup; at d=1024 that is a guess,
# and the failure it guards against (divergence, or a flat arm) would cost the night.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128
until grep -q "XL READY" runs/prep_xl.log 2>/dev/null; do sleep 30; done
LOG=runs/lr_probe_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --Mc 256 --dv 256 --Ls 64 --ff 4096 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --Md 4 --G 8 --freq rope
        --variant lapa_cc --amp bf16 --warmup 100
        --seconds 420 --eval-batches 30 --eval-every 140
        --samples 0 --only sca2 --log $LOG"
for LR in 1e-3 5e-4 3e-4; do
  echo "##### $(date +%H:%M) lr=$LR"
  python -u pretrain.py --label "lr$LR" --seed 0 $COMMON --lr $LR
done
echo "##### LR_PROBE DONE"
