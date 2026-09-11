#!/usr/bin/env bash
# LapA v1 (mixed grid + window 64, M=190) at 12000 steps: equal-budget row against GDN 3x80 (12k).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 12000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
echo "##### $(date +%H:%M) LapA v1, 12000 steps"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA v1 mixed + window 64 (12k steps)" --rope-base 1000 --slow-frac 0.25 --L 64 --save runs/ck_copy_v1_12k.pt
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY D128 V1 12K DONE"
