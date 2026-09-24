#!/usr/bin/env bash
# T=2048, 16h: dv=384 ff=5800 + conv_silu (silu after the causal conv).
# The entire path from z to the state is LINEAR in our architecture.
# GDN puts silu on q, k, v BEFORE they enter the state — the values stored
# are non-linear in z. --conv-silu adds silu after the conv, so z passes
# through a non-linearity before entering K, V, and both heads. Zero params.
# Tested once at 5h before gdn-gate (d1024_convsilu, null), but the gate
# changes the context: silu at input + silu at output = two non-linear
# points, much closer to GDN's five.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_t2048_dv384\.t2048_dv384\.pt" runs/long16h_t2048.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) t2048_dv384 finished"

LOG=runs/long16h_t2048.jsonl
COMMON="--data pycode_long2048_xl.pt --block 2048 --batch 8 --d 1024 --layers 8
        --ff 5800 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 57600 --eval-batches 60 --eval-every 600 --pos-buckets 16
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 7200
        --variant lapa_cc --Mc 256 --dv 384 --Ls 128 --theta-scale 0.02
        --rope-base 2048 --slow-frac 0.25 --conv 4 --conv-silu"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell t2048_dv384_silu --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate
echo "##### DONE"
