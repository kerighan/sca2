#!/usr/bin/env bash
# GDN at 12000 steps: does its cliff at L>=256 move with more training, or is it capacity?
# Same shape as the 4000-step arm (3x80, ff 260, state 21.4k ~ LapA M=190 window 16). Waits for w32.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "COPY D128 W32 DONE" runs/copy_d128_w32.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 12000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl"
echo "##### $(date +%H:%M) GDN 3x80, 12000 steps"
python -u -m lapa.benchmarks.copy run $C --arm gdn --label "GDN 3x80 (12k steps)" --gdn-heads 3 --gdn-head-k 80 --ff 260 --save runs/ck_copy_gdn_12k.pt
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY D128 GDN12K DONE"
