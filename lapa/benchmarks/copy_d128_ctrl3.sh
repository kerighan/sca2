#!/usr/bin/env bash
# Structural fix for the exactness floor: the SHORT head's Dirichlet comb is an exact tap at
# every lag < L_window. A copy of length L is a read at lag L+1, so a 64-token window serves
# L <= 62 with zero interference; the long head only takes over beyond. Two arms: M=190 and
# M=380 with --L 64 (state +4.9k floats/layer). Waits for ctrl2.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "COPY D128 CTRL2 DONE" runs/copy_d128_ctrl2.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 4000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl"
run () { echo "##### $(date +%H:%M) $*"; python -u -m lapa.benchmarks.copy run $C "$@"; }
run --arm lapa --label "LapA M=190 window 64" --M 190 --dv 56 --ff 448 --L 64
run --arm lapa --label "LapA M=380 window 64" --M 380 --dv 56 --ff 448 --L 64
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
echo "##### COPY D128 CTRL3 DONE"
