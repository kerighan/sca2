#!/usr/bin/env bash
# Mamba2 baseline on pycode, same conditions as all other arms.
# expand=1 -> 11.8M/layer (between LapA 10.8M and GDN 13.7M)
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_mamba2 --variant mamba2_cc --mamba-expand 1
echo "##### MAMBA2 DONE"
