#!/usr/bin/env bash
# gen3 with the C read UNNORMALISED (cdelta_raw), longrun protocol, class breakdown.
# CATCHUP.md: the C head hurts word_new (+0.28, gone at Mc=2); the proposed cause is
# that _rms() erases the read's magnitude, i.e. the only evidence of "nothing
# matched". One change vs catch_gen3_s0 (+112 params/layer, 186071 vs 185959).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
SHAPE="--Mc 190 --dv 56 --ff 364 --theta-scale 0.02"
echo "##### catch_raw_s0 :: --seed 0 --variant cdelta_raw_cc $SHAPE"
python -u pretrain.py --label catch_raw_s0 --seed 0 --variant cdelta_raw_cc $COMMON $SHAPE --save runs/ck_catch_raw_s0
echo "##### LONG_RAW DONE"
