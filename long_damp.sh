#!/usr/bin/env bash
# gen3 with learned per-mode decay on the long head (cdelta_damp, sca2/arch_damp.py).
# ff=364 -> 186149 params (+0.10% vs gen3). Longrun protocol, class breakdown. NOT
# launched automatically: queue it only if the short/verification ablation leaves
# room (user's call). Single change vs catch_gen3_s0.
#   setsid -f nohup bash long_damp.sh > runs/long_damp.log 2>&1 < /dev/null
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
echo "##### catch_damp_s0 :: --seed 0 --variant cdelta_damp_cc"
python -u pretrain.py --label catch_damp_s0 --seed 0 --variant cdelta_damp_cc $COMMON --Mc 190 --dv 56 --ff 364 --theta-scale 0.02 --save runs/ck_catch_damp_s0
echo "##### LONG_DAMP DONE"
