#!/usr/bin/env bash
# 10h run: gdngate + Ls=128 + M=512 dv=512 (bigger mixer, same FFN).
# The FFN arm was WORSE than base — gs_ff already shows the FFN is oversized.
# This puts params in the mixer instead: 139.6M total (-10.8% vs GDN).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long10h_d1024.jsonl

COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 512 --dv 512 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell l10_gdngate_M512 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### LONG10H_BIGMIX DONE"
