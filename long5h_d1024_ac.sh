#!/usr/bin/env bash
# d=1024: FOUR read groups + Triton scan kernel.
#
# NG=4 with triton_scan runs at 244k tok/s on the layer (fwd+bwd) — the SAME
# speed as NG=1 batched (243k). The groups are FREE thanks to the fused kernel.
# NG=2 with triton_scan is 263k tok/s, faster than anything batched.
#
# So the question "does NG=4 help the loss" can now be asked WITHOUT paying
# a speed tax. If it does, we have 4 temporal profiles per layer instead of 2,
# at no cost — structural capacity for free.
#
# Iso gate: float64 3.9e-16, float32 1.8e-07 (--self, NG=4, triton_scan).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_g4tri --layer-scale --lam-free --damp-mem 4,20000 --long-groups 4
echo "##### LONG5H_D1024_AC DONE"
