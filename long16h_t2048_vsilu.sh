#!/usr/bin/env bash
# T=2048, 16h: dv=384 ff=5800 + v_silu (silu on the values, not the conv).
# GDN does v = silu(conv(Wv @ z)) — the values stored are non-linear in z.
# We do v = V(z) — purely linear. --v-silu adds silu(V(z)) so the state
# captures non-linear features. Zero params, one elementwise op.
# No conv_silu — the non-linearity is specifically on the values, not on z.
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

cell t2048_vsilu --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate --v-silu
echo "##### DONE"
