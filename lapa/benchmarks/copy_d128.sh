#!/usr/bin/env bash
# Local copy-capacity run (RTX 2070, ~25 min per arm). Two questions at d=128:
#  (1) does the copy cliff track M?  LapA M = 64 / 128 / 190 / 380 (dv fixed)
#  (2) LapA vs GDN at matched STATE: LapA M=95 (~12k floats) ~ GDN 3x60 (12.4k);
#      LapA M=190 (22.7k) ~ GDN 3x80 (21.6k). Attention = ceiling (carries the prefix).
# Lengths 32..512 -> Tmax 1026, the training shape of the campaign.
#   setsid -f nohup bash lapa/benchmarks/copy_d128.sh > runs/copy_d128.log 2>&1 < /dev/null
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/../.."
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --steps 4000 --lr 1e-3
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl"
run () { echo "##### $(date +%H:%M) $*"; python -u -m lapa.benchmarks.copy run $C "$@"; }
run --arm lapa --label "LapA M=190 (22.7k)" --M 190 --dv 56 --ff 448
run --arm gdn  --label "GDN 3x80 (21.6k)"   --gdn-heads 3 --gdn-head-k 80 --ff 260
run --arm lapa --label "LapA M=95 (12.1k)"  --M 95  --dv 56 --ff 448
run --arm gdn  --label "GDN 3x60 (12.4k)"   --gdn-heads 3 --gdn-head-k 60 --ff 260
run --arm lapa --label "LapA M=380 (44k)"   --M 380 --dv 56 --ff 448
run --arm lapa --label "LapA M=64 (8.6k)"   --M 64  --dv 56 --ff 448
run --arm attn --label "attention (KV cache)" --attn-heads 4 --ff 448
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128_exact.png --exact
echo "##### COPY D128 DONE"
