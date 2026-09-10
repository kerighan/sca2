#!/usr/bin/env bash
# One seed of cdelta_bp (content phase bounded to +-pi/2, CATCHUP.md conjecture 1)
# under EXACTLY the longrun.sh protocol, so it reads against long_gen3_s0 and
# long_gdn_s0 point for point: same corpus, seed, shape, budget, eval schedule.
# The question is not the endpoint (n=1 cannot resolve 0.05) but the SHAPE:
# does the gap to GDN still cross at ~300M, and does the word_new class (added
# via --class-eval, free) close slower than in gen3.
#   setsid nohup bash long_bp.sh > runs/long_bp.log 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/longrun.jsonl --class-eval"
SHAPE="--Mc 190 --dv 56 --ff 364 --theta-scale 0.02"
echo "##### long_bp_s0 :: --seed 0 --variant cdelta_bp_cc $SHAPE"
python -u pretrain.py --label long_bp_s0 --seed 0 --variant cdelta_bp_cc $COMMON $SHAPE --save runs/ck_long_bp_s0
echo "##### LONG_BP DONE"
