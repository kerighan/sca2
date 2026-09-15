#!/usr/bin/env bash
# d=1024: GDN-style readout gate on the long head.
#
# THE HYPOTHESIS: the gap to GDN is not in the memory, not in the temporal
# diversity, not in the addressing — it is in the OUTPUT PROCESSING. Our gate
# is a SCALAR (sigmoid of a cosine match), GDN's is a PER-CHANNEL non-linear
# function of x (silu(Linear(d, value_dim))). Three campaigns of temporal
# enrichment (lam_free, long_groups, init tricks) were analytically correct
# and empirically null. The three mechanisms that DID pay (conv, kv gate,
# LayerScale) all act on processing, not memory.
#
# WHAT THIS ARM DOES: replaces _out from
#   RMS(val) * sigmoid(ga * cos(key_read, Kv(z)) + gb)     <- 1 number, all channels
# to
#   LayerNorm(val) * silu(Linear(d -> 2*dv)(z))             <- per channel, non-linear
#
# Cost: +4,075,504 params per arm (86.7M vs 82.6M), still 21% BELOW GDN's
# 109.4M so parameter matching is not a concern. kv_dk is dropped (no cosine
# match needed), so dvi = dv = 256 instead of 272, which slightly SHRINKS
# the state.
#
# NG=1, no long_groups — isolate the gate change. triton_scan for speed.
# --lam-free and --layer-scale carried over from lsfree (our best NG=1 arm).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_d1024_big_g2" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_big_g2 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_gdngate --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### LONG5H_D1024_AE DONE"
