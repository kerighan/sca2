#!/usr/bin/env bash
# Tomorrow's LM arm: MIXED grid at M=190, window 16 -- 25% slow integrators (periods 2T..20T, what the
# base-1e4 LM used for 55-85% of its state energy) + 75% dense rope at base 1e3 (the copy bench's
# addressing win). Single change vs A (l5_A_s0). Waits for tonight's rope queue to finish.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
until grep -q "LONG5H ROPE DONE" runs/long5h_rope.log 2>/dev/null; do sleep 120; done
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log runs/long5h.jsonl --class-eval"
echo "##### $(date +%H:%M) l5_Amixed_s0"
python -u pretrain.py --label l5_Amixed_s0 --seed 0 --variant cshort_damph_cc $COMMON --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25 --save runs/ck_l5_Amixed_s0
echo "##### LONG5H MIXED DONE"
