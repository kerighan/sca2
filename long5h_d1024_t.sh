#!/usr/bin/env bash
# d=1024: spend the architecture's advantage on DEPTH instead of speed.
#
# LapA's mixer is linear in d, GDN's is quadratic -- 10.32M/layer against 13.67M. That is
# the structural claim this architecture has rested on since SPARK.md §3(a), and the whole
# campaign has cashed it as THROUGHPUT and never as capacity. At GDN's per-layer budget we
# can afford 10 layers where it has 8:
#
#   LapA 8 layers  116.1M  -18.7% vs GDN-8   24.6k tok/s   442M tokens in 5 h
#   LapA 10 layers 136.8M   -4.3%            ~19.7k        ~354M
#   GDN  8 layers  142.9M    ref             21.1k         381M
#
# So this arm is still the SMALLER model (-4.3%), and depth is generally the better buy
# per parameter than width. Nothing else changes: the base is d1024_fast_kv_conv4, the
# best configuration to date (+0.046 from GDN at equal wall clock).
#
# THE TRADE, stated before the result so it cannot be quoted selectively: at 10 layers we
# lose the throughput advantage AND land behind GDN on tokens (354M vs 381M). This arm has
# to be read at MATCHED TOKENS, and it will look worse at equal wall clock than the 8-layer
# base does. That is the opposite of the trade every previous arm made, and it is the point
# -- we have never once tested whether the cheap layer is worth more as an extra layer than
# as extra speed.
#
# Also queued: --conv 8. The conv did half the work of closing the gap to GDN and its
# kernel was never tuned -- k=4 was copied from GDN's q/k/v convs. And d1024_sg4 just told
# us something about it: its gain is NOT per-channel diversity of the READ (else grouping
# the short head's taps would have paid, and it came in at -0.007 over 39 points). The conv
# is depthwise so it does not mix channels either. What it does that nothing else does is
# shape z BEFORE both heads -- what gets WRITTEN, not just what gets read. Widening its
# window is the cheapest probe of that.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_dv512" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_dv512 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_L10   --layers 10 --conv 4
cell d1024_conv8 --layers 8  --conv 8
echo "##### LONG5H_D1024_T DONE"
