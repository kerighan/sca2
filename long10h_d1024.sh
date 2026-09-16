#!/usr/bin/env bash
# 10h run: our best config (gdngate + Ls=128 + triton_scan) vs GDN.
# At 5h the gap to GDN was +0.008 at matched tokens and negative in wall-clock.
# The noise floor drops from sd=0.065 (100-200M) to 0.022 (200M+) — the second
# half of a 5h run is where conclusions become reliable, so 10h doubles the
# reliable zone and lets us see if the trend continues.
#
# Two arms, sequential: LapA first (10h), then GDN (10h). Total: 20h.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128

LOG=runs/long10h_d1024.jsonl
: > "$LOG"

COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

# ARM 1: LapA best (gdngate + Ls=128 + triton_scan)
SCA2_LONG_PATH=triton_scan cell l10_gdngate \
    --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02 \
    --rope-base 1000 --slow-frac 0.25 --conv 4 \
    --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate

# ARM 2: GDN (Triton fla kernels)
cell l10_gdn \
    --variant gdn_cc --gdn-heads 8 --gdn-head-k 128 --gdn-expand-v 1.0

echo "##### LONG10H DONE"
