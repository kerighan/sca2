#!/usr/bin/env bash
# One arm per hypothesis, all at M=190 (user's rule), sequential on a free GPU:
#   window 64      short head's exact comb covers copies up to L=62
#   no delta rule  beta frozen at 0 (additive write, as the old head)
#   rope base 1e3 / 1e5   the long head's grid: a smaller base packs the frequencies
#                  toward high omega (sharper short lags, aliases sooner), a larger one
#                  spreads them (longer unaliased range, broader kernel)
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 4000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
run () { echo "##### $(date +%H:%M) $*"; python -u -m lapa.benchmarks.copy run $C --arm lapa "$@"; }
run --label "LapA M=190 window 64"    --L 64
run --label "LapA M=190 no delta rule" --beta-init -1000
run --label "LapA M=190 rope 1e3"     --rope-base 1000
run --label "LapA M=190 rope 1e5"     --rope-base 100000
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png --hide "theta=0,no damping,M=64,12k"
echo "##### COPY D128 NEXT DONE"
