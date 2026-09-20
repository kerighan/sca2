#!/usr/bin/env bash
# 10h: dv=384 ff=5800, param-matched against GDN (+0.3%).
# The missing insight from the dv512 checkpoint: gs_ff went ABOVE 1.0,
# meaning the FFN was too small for the wider mixer. dv and ff must scale
# together. dv=384 ff=5800 keeps the same mix/FFN ratio as the base (~28%)
# while matching GDN at 157.1M total.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_t2048_gdn\.t2048_gdn\.pt" runs/long16h_t2048.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) t2048_gdn finished"

LOG=runs/long10h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 5800 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 384 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell l10_dv384_ff5800 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### DONE"
