#!/usr/bin/env bash
# Day 3 morning queue (GPU idle): (1) copy, mixed grid + window 64 at 8000 steps -- is the 4000-step
# gap to rope1e3+w64 slower convergence or a plateau; (2) LM 5 h XL: the candidate layer
# mixed grid + window 64 (ff 424), to have it measured where it matters.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
cd "$(dirname "$0")"
C="--d 128 --layers 2 --lengths 32,64,128,256,512 --symbols 64 --batch 32 --lr 1e-3 --steps 8000
   --eval-every 250 --eval-batches 4 --eval-batch 64 --compile --log runs/copy_d128.jsonl --M 190 --dv 56 --ff 448"
echo "##### $(date +%H:%M) copy mixed + w64, 8000 steps"
python -u -m lapa.benchmarks.copy run $C --arm lapa --label "LapA M=190 mixed grid + window 64 (8k steps)" --rope-base 1000 --slow-frac 0.25 --L 64 --save runs/ck_copy_mixed_w64_8k.pt
python -m lapa.benchmarks.copy plot runs/copy_d128.jsonl --out plot/copy_d128.png
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 128 --layers 4 --Md 4 --G 8 --freq rope
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8 --samples 0 --only sca2 --log runs/long5h.jsonl --class-eval"
echo "##### $(date +%H:%M) l5_Amixedw64_s0"
python -u pretrain.py --label l5_Amixedw64_s0 --seed 0 --variant cshort_damph_cc $COMMON --Mc 190 --dv 56 --ff 424 --Ls 64 --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25 --save runs/ck_l5_Amixedw64_s0
echo "##### DAY3 QUEUE DONE"
