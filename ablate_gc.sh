#!/bin/bash
# Gated C head ablations: 900 s each, same data/order/seed, 2 layers, Mc=128 Md=16.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
C="--only sca2 --Mc 128 --Md 16 --seconds 900 --eval-every 60 --layers 2 --data fineweb_300M.pt --log runs/gc_ablate.jsonl --samples 0"
run() { echo "=== $(date +%H:%M) $1 ==="; shift; python -u pretrain.py $C "$@"; }
run base        --variant v3polar_cc --label base
run decay       --variant gc_cc --c-decay --label decay
run decay_h4    --variant gc_cc --c-decay --c-heads 4 --label decay_h4
run decay_h4_kq --variant gc_cc --c-decay --c-heads 4 --c-sepq --label decay_h4_kq
run conv        --variant gc_cc --c-decay --c-heads 4 --c-sepq --conv 4 --label decay_h4_kq_conv
run full        --variant gcfull_cc --label full
echo "=== $(date +%H:%M) ALL DONE ==="
