#!/usr/bin/env bash
# T=2048, 16h: dv=128 Mc=128, same everything else.
# Third point of the mixer-size sweep: 128 / 256 / 384.
# 111.3M total, -28.9% vs GDN. If wall-clock parity holds here too, the
# mixer is oversized at 256 and the speed gain is free.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long16h_t2048.jsonl
COMMON="--data pycode_long2048_xl.pt --block 2048 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 57600 --eval-batches 60 --eval-every 600 --pos-buckets 16
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 7200
        --variant lapa_cc --Mc 128 --dv 128 --Ls 128 --theta-scale 0.02
        --rope-base 2048 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell t2048_dv128 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate --v-silu --init-v2
echo "##### DONE"
