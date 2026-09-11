#!/usr/bin/env bash
# M=256 on the v1 layer (mixed grid + window 64): copy at 8k steps, then the 5 h XL LM arm
# with ff 388 so params stay matched to GDN 3x80 (185797 vs 185714). Waits for the v1 12k copy run.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
cd "$(dirname "$0")"
until grep -q "COPY D128 V1 12K DONE" runs/copy_d128_v1_12k.log 2>/dev/null; do sleep 60; done
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 8000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --dv 56 --ff 448"
echo "##### $(date +%H:%M) copy v1 M=256, 8000 steps"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA v1 M=256 mixed + window 64 (8k steps)" --M 256 --rope-base 1000 --slow-frac 0.25 --L 64 --save runs/ck_copy_v1_m256_8k.pt
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 128 --layers 4 --Md 4 --G 8 --freq rope
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8 --samples 0 --only sca2 --log runs/long5h.jsonl --class-eval"
echo "##### $(date +%H:%M) l5_v1m256_s0"
python -u pretrain.py --label l5_v1m256_s0 --seed 0 --variant cshort_damph_cc $COMMON --Mc 256 --dv 56 --ff 388 --Ls 64 --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25 --save runs/ck_l5_v1m256_s0
echo "##### M256 QUEUE DONE"
