#!/usr/bin/env bash
# The two winning knobs together, still one arm at M=190: rope base 1e3 + short window 64.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "COPY D128 WIN DONE" runs/copy_d128_win.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 4000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
echo "##### $(date +%H:%M) rope 1e3 + window 64"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA M=190 rope 1e3 + window 64" --rope-base 1000 --L 64
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY D128 COMBO DONE"
