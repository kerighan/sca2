#!/usr/bin/env bash
# A with the decay cap lifted (cshort_damphf: lambda <= 60/CTX = 0.47, memories down to
# ~2 tokens). In catch_shortdamp_s0 the damped half of the modes ran into the 1/8 cap
# (median memory 8 tokens, 71/95 modes at the cap in layers 0-1). Single change vs A.
# NOT launched automatically; queue after B (catch_shortdampkv_s0) if wanted:
#   setsid -f nohup bash long_fast.sh > runs/long_fast.log 2>&1 < /dev/null
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
until grep -q "DAMP QUEUE DONE" runs/long_damp_queue.log 2>/dev/null; do sleep 60; done
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
echo "##### catch_shortfast_s0 :: --seed 0 --variant cshort_damphf_cc"
python -u pretrain.py --label catch_shortfast_s0 --seed 0 --variant cshort_damphf_cc $COMMON --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02 --save runs/ck_catch_shortfast_s0
echo "##### LONG_FAST DONE"
