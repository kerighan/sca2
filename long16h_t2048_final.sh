#!/usr/bin/env bash
# T=2048, 16h: dv=384 ff=5800 with and without conv_silu.
# Triton kernel optimized for D=384 by the kernel agent (80.4 ms, +10%).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long16h_t2048.jsonl

COMMON="--data pycode_long2048_xl.pt --block 2048 --batch 8 --d 1024 --layers 8
        --ff 5800 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 57600 --eval-batches 60 --eval-every 600 --pos-buckets 16
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 7200
        --variant lapa_cc --Mc 256 --dv 384 --Ls 128 --theta-scale 0.02
        --rope-base 2048 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

# ARM 1: dv=384 ff=5800 (param-matched GDN, no silu)
cell t2048_dv384v2 --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate

# ARM 2: same + conv_silu (non-linearity before the state)
cell t2048_dv384_silu --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate --conv-silu

echo "##### T2048 FINAL DONE"
