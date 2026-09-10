#!/usr/bin/env bash
# Seed 1 of A (catch_shortdamp): turns the -0.034 lead over GDN from an observation
# into a result -- the long run's gaps reproduced across seeds to 0.001. Paired
# against long_gdn_s1 (same seed, same corpus) from runs/longrun.jsonl.
# Waits for long_fast.sh to finish.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
until grep -q "LONG_FAST DONE" runs/long_fast.log 2>/dev/null; do sleep 60; done
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
echo "##### catch_shortdamp_s1 :: --seed 1 --variant cshort_damph_cc"
python -u pretrain.py --label catch_shortdamp_s1 --seed 1 --variant cshort_damph_cc $COMMON --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02 --save runs/ck_catch_shortdamp_s1
echo "##### LONG_A_S1 DONE"
