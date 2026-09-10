#!/usr/bin/env bash
# Two arms, longrun protocol, class breakdown (CATCHUP.md results section):
#  1. catch_noc_s0  gen3 with the C head crippled (Mc=2: two modes cannot retrieve),
#                   ff=460 to match (186003 vs 185959). If the word_new deficit
#                   (+0.28 uniform over frequency and position, in every arm that has
#                   the C head) disappears here, the C head actively HURTS non-retrieval
#                   tokens; if it persists, SCA2 is missing a capability GDN has.
#  2. catch_hyb_s1  the hybrid's second seed: parity with GDN at 500M (-0.014) needs n=2.
#   setsid nohup bash queue_night.sh > runs/queue_night.log 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON --save runs/ck_$label "$@"; }
cell catch_noc_s0 --seed 0 --variant cdelta_cc --Mc 2 --dv 56 --ff 460 --theta-scale 0.02
cell catch_hyb_s1 --seed 1 --variant hyb_cc --Mc 190 --dv 56 --ff 244 --theta-scale 0.02 --gdn-heads 2 --gdn-head-k 48 --gdn-expand-v 1.0
echo "##### QUEUE DONE"
