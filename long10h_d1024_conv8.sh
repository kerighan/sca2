#!/usr/bin/env bash
# 10h: gdngate + Ls=128 + conv=8 (double the causal conv window).
# Conv=4 was the single biggest gain of the whole campaign alongside the
# gdn-gate. Doubling it to 8 costs +32k params (nothing) and lets the
# pre-processing see 8 tokens instead of 4. This is an EFFICIENCY lever,
# not a capacity lever — exactly the kind that has paid every time.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long10h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 8"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell l10_conv8 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### DONE"
