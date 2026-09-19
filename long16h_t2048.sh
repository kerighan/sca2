#!/usr/bin/env bash
# T=2048, 16h per arm: the long-context test.
#
# At T=1024, 82% of the loss drop happens in the first 512 tokens, and every
# architecture (ours, GDN, Mamba) follows the same curve — no intervention
# on temporal diversity, capacity, or processing changes it. The hypothesis:
# T=1024 doesn't exercise long-range retrieval enough to differentiate.
#
# T=2048 doubles the context. Documents that need 1000+ tokens of context
# (function calls referencing imports at the top, variables defined hundreds
# of lines earlier) now fit in the window. If our spectral memory has an
# advantage over GDN's matrix memory, it shows here.
#
# rope_base=2048: the rule is 2*base ~ context. slow_frac=0.25 puts 25% of
# modes on periods 4096..40960 tokens, so the slow integrators now see the
# WHOLE context (periods 2x-20x the window), not 2x-20x beyond it.
#
# 1519.6M train tokens. At 26k tok/s, 16h = ~1 epoch. At 20k (GDN), 0.8.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128

LOG=runs/long16h_t2048.jsonl
: > "$LOG"

COMMON="--data pycode_long2048_xl.pt --block 2048 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 57600 --eval-batches 60 --eval-every 600 --pos-buckets 16
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 7200"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

# ARM 1: LapA best (gdngate + Ls=128 + triton_scan)
SCA2_LONG_PATH=triton_scan cell t2048_gdngate \
    --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02 \
    --rope-base 2048 --slow-frac 0.25 --conv 4 \
    --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate

# ARM 2: GDN (Triton fla kernels)
cell t2048_gdn \
    --variant gdn_cc --gdn-heads 8 --gdn-head-k 128 --gdn-expand-v 1.0

echo "##### T2048 DONE"
