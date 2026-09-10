#!/usr/bin/env bash
# User's order: short-head window 64 first (the exact comb covers copies up to L=62), then the
# delta-rule control. Waits for the no-damping arm's python (PID 0) to exit.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
while kill -0 0 2>/dev/null; do sleep 15; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 4000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl"
run () { echo "##### $(date +%H:%M) $*"; python -u -m lapa.benchmarks.copy run $C "$@"; }
run --arm lapa --label "LapA M=190 window 64" --M 190 --dv 56 --ff 448 --L 64
run --arm lapa --label "LapA M=380 window 64" --M 380 --dv 56 --ff 448 --L 64
run --arm lapa --label "LapA M=190 no delta rule" --M 190 --dv 56 --ff 448 --beta-init -1000
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png --hide "theta=0,no damping,M=64"
echo "##### COPY D128 NEXT DONE"
