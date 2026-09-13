#!/usr/bin/env bash
# d=1024: depth at matched parameters, then the erase gate's init.
#
# Waits for d1024_convsilu to save. SiLU came in at a median +0.009 against the base over
# 39 spike-filtered points -- null, marginally worse. It did catch up from a +0.167 start,
# which is the identity-at-init penalty being absorbed exactly as predicted, but catching
# up to parity is not winning.
#
# ARM 1  --layers 10. The architecture's structural claim, finally spent on capacity
#   instead of speed: LapA is 10.32M/layer against GDN's 13.67M because its mixer is linear
#   in d where GDN's is quadratic, so 10 layers cost 136.8M against GDN-8's 142.9M -- still
#   4.3% SMALLER. Read it at MATCHED TOKENS: it drops to ~19.7k tok/s, so it lands behind
#   GDN on tokens too (354M vs 381M) and will look worse at equal wall clock than the
#   8-layer base. That is the opposite of every trade this campaign has made, which is the
#   point -- nobody has checked whether the cheap layer is worth more as an extra layer
#   than as extra speed.
#
# ARM 2  --beta-init 0.0. GDN's b projection has NO BIAS, so its erase gate starts at
#   sigmoid(0) = 0.5; ours starts at sigmoid(-2) = 0.12, four times more timid. And the
#   checkpoint ablations put GDN's delta rule at +4.51 nats when disabled against our +1.42
#   -- the same nominal mechanism carrying three times the load. An init that throttles the
#   error-correcting write from step one is a direct candidate for part of that. Free: one
#   flag, no parameters, no throughput.
#
# A caveat on reading either of them. The last four arms -- kv_conv4 +0.101, its exact port
# +0.114, sg4 +0.103, convsilu +0.133 -- all sit within 0.03 of each other, and two of them
# compute THE SAME FUNCTION and differ by 0.013. Single 5 h arms cannot resolve below about
# +-0.03 on the median. Anything smaller than that needs 10 h or a second seed, not another
# 5 h arm.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_convsilu" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_convsilu finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_L10       --layers 10
cell d1024_beta0     --layers 8 --beta-init 0.0
echo "##### LONG5H_D1024_V DONE"
