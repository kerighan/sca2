#!/usr/bin/env bash
# Controls for the exactness floor seen on LapA M=190 (0.99/0.78 at L=32 vs GDN 1.00/0.93):
#   theta=0      purely positional addressing (the old bench's setting; content phase can only add noise on random symbols)
#   persist=1    no damping (all modes lambda=0): does forgetting cost exact copy?
#   12000 steps  the old bench's budget, is it just undertrained?
# Waits for copy_d128.sh to finish.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "COPY D128 DONE" runs/copy_d128.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl"
run () { echo "##### $(date +%H:%M) $*"; python -u -m lapa.benchmarks.copy run $C "$@"; }
run --arm lapa --label "LapA M=190 theta=0"     --M 190 --dv 56 --ff 448 --theta-scale 0 --steps 4000
run --arm lapa --label "LapA M=190 no damping"  --M 190 --dv 56 --ff 448 --persist 1.0 --steps 4000
run --arm lapa --label "LapA M=190 12k steps"   --M 190 --dv 56 --ff 448 --steps 12000
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128_exact.png --exact
echo "##### COPY D128 CTRL DONE"
