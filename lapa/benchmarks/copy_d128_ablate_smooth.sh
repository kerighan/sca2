#!/usr/bin/env bash
# Which of the two smoothing changes stabilised the copy run? Two single-change arms against
# the plain v1 8k run (old damping, constant lr) and the smooth one (both changes):
#   damping only : window-aligned damping, constant lr 1e-3, no warmup   <- "brutal" mode
#   schedule only: old damping (lam_max 0.125, mem 64..4096) + warmup 500 + cosine
# Waits for the M=256 LM arm (5 h).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "M256 QUEUE DONE" runs/queue_m256.log 2>/dev/null; do sleep 120; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 8000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl
   --M 190 --dv 56 --ff 448 --rope-base 1000 --slow-frac 0.25 --L 64"
echo "##### $(date +%H:%M) damping only (aligned damping, constant lr)"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA v1 aligned damping only (8k)"
echo "##### $(date +%H:%M) schedule only (old damping + warmup/cosine)"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA v1 warmup+cosine only (8k)" --lam-max 0.125 --damp-mem 64,4096 --warmup 500 --cosine
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY ABLATE SMOOTH DONE"
