#!/usr/bin/env bash
# Window sweep at M=190 (one arm at a time), now that the banded read makes wide windows cheap:
# does the receptive field keep composing across the two layers (2 x (L-1))?
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "COPY D128 NEXT DONE" runs/copy_d128_next.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 4000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
run () { echo "##### $(date +%H:%M) $*"; python -u -m lapa.benchmarks.copy run $C --arm lapa "$@"; }
run --label "LapA M=190 window 128" --L 128
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png --hide "theta=0,no damping,M=64,12k"
echo "##### COPY D128 WIN DONE"
