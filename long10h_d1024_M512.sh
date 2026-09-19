#!/usr/bin/env bash
# 10h: gdngate + Ls=128 + M=512 (dv=256 fixe). Isoler l'effet de M.
# +2.1M params (122.9M total), quasi gratuit. Already null at 5h without
# gdn-gate (d1024_lapa_M512), but the gate changes what M buys.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_l10_dv512\.l10_dv512\.pt" runs/long10h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) l10_dv512 finished"

LOG=runs/long10h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 512 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell l10_M512 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### DONE"
