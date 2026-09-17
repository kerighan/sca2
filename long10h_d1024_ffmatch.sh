#!/usr/bin/env bash
# 10h run: gdngate + Ls=128 + ff=5196 (param-matched against PLE).
# Queued after l10_gdngate_ple. Same JSONL.
#
# PLE adds 17.8M params via per-layer embeddings (token identity shortcut).
# This arm adds the same 17.8M via a bigger FFN (ff 4096 -> 5196).
# If PLE wins: the deep layers needed token identity, not more processing.
# If FFN wins: the layers just needed more capacity, the identity signal was noise.
# If tied: the extra params help the same amount regardless of where they go.
#
# total: 138,782,328 (vs PLE 138,576,920, diff +205k = 0.15%)
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_l10_gdngate_ple\.l10_gdngate_ple\.pt" runs/long10h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) l10_gdngate_ple finished"

LOG=runs/long10h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 5196 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell l10_gdngate_ff --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### LONG10H_FFMATCH DONE"
