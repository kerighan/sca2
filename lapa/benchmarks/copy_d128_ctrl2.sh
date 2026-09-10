#!/usr/bin/env bash
# Fourth control: the DELTA RULE. The old additive head (v3polarflat, M=32) copied L=32 at
# 0.98 exact; LapA writes e = v - beta.vhat with beta=0.12 at init, and the rope Gram has
# kappa[1]=0.81, so every write erases ~10% of its neighbour's value. beta_init=-1000 ->
# sigmoid = 0 with zero gradient: the delta rule is OFF, the write is additive.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
until grep -q "COPY D128 CTRL DONE" runs/copy_d128_ctrl.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl"
echo "##### $(date +%H:%M) no delta rule"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA M=190 no delta rule" --M 190 --dv 56 --ff 448 --beta-init -1000 --steps 4000
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128_exact.png --exact
echo "##### COPY D128 CTRL2 DONE"
