#!/usr/bin/env bash
# Copy bench, the candidate layer: MIXED grid (25% slow integrators + dense base 1e3) + window 64,
# M=190. Reference arms already in runs/copy_d128.jsonl: rope 1e3 + window 64 (the copy champion),
# window 64 alone, base. Question: does keeping the LM's slow quarter cost copy capacity?
# Waits for the mixed-grid LM arm to finish (~06:30).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "LONG5H MIXED DONE" runs/long5h_mixed.log 2>/dev/null; do sleep 120; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 4000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
echo "##### $(date +%H:%M) mixed grid + window 64"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA M=190 mixed grid + window 64" --rope-base 1000 --slow-frac 0.25 --L 64 --save runs/ck_copy_mixed_w64.pt
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY D128 MIXED DONE"
