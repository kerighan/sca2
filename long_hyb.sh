#!/usr/bin/env bash
# HYBRID under the longrun.sh protocol: cdelta C head + GDN head (2 x 48, expand 1)
# in the D slot, ff=244 to match parameters (186003 vs champion 185959, +0.02%;
# vs GDN 185714, +0.16%). Reads point for point against catch_gen3_s0 and
# catch_gdn_s0 (same corpus, seed, budget, eval schedule, class breakdown).
# See sca2/arch_hybrid.py and CATCHUP.md.
#   setsid nohup bash long_hyb.sh > runs/long_hyb.log 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
SHAPE="--Mc 190 --dv 56 --ff 244 --theta-scale 0.02 --gdn-heads 2 --gdn-head-k 48 --gdn-expand-v 1.0"
echo "##### catch_hyb_s0 :: --seed 0 --variant hyb_cc $SHAPE"
python -u pretrain.py --label catch_hyb_s0 --seed 0 --variant hyb_cc $COMMON $SHAPE --save runs/ck_catch_hyb_s0
echo "##### LONG_HYB DONE"
