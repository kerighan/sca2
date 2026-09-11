#!/usr/bin/env bash
# Window 32 on the candidate's grid (mixed: 25% slow + base 1e3), M=190, 8000 steps like the
# window-64 run it compares to. Waits for the day-3 queue (the candidate's LM run) to finish.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "DAY3 QUEUE DONE" runs/queue_day3.log 2>/dev/null; do sleep 120; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 8000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
echo "##### $(date +%H:%M) mixed grid + window 32, 8000 steps"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA M=190 mixed grid + window 32 (8k steps)" --rope-base 1000 --slow-frac 0.25 --L 32 --save runs/ck_copy_mixed_w32_8k.pt
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY D128 W32 DONE"
