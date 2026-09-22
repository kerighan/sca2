#!/usr/bin/env bash
# T=2048, 16h: dv=256 ff=4096 + init_v2 + v_silu + k_silu.
# Same as initv2 but at the BASE size (120.7M, 23% fewer params than GDN).
# k_silu adds non-linearity on the keys — richer content features for the
# phase computation, like GDN's q = silu(conv(Wq @ z)).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long16h_t2048.jsonl
COMMON="--data pycode_long2048_xl.pt --block 2048 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 57600 --eval-batches 60 --eval-every 600 --pos-buckets 16
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 7200
        --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 2048 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell t2048_initv2_kv --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate --v-silu --k-silu --init-v2
echo "##### DONE"
