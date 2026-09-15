#!/usr/bin/env bash
# d=1024: gdngate + Ls=128 (double the short-head window).
#
# Ls has NEVER been ablated at this scale. The short head is an exact L-tap
# causal DFT filter: it sees precisely Ls tokens, no more, no less. At Ls=64
# that covers 6.25% of the T=1024 context. At Ls=128 it covers 12.5%.
#
# This is a much simpler hypothesis than long_groups: maybe we just need the
# short head to see more local context. Cost: +525k params (0.6%), negligible.
#
# Base: gdngate (our best wall-clock arm, +0.9% vs GDN at equal time).
# Single change: --Ls 128 instead of 64.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_d1024_big_gdngate" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_big_gdngate finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_gdngate_L128 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### LONG5H_D1024_AG DONE"
