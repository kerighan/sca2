#!/usr/bin/env bash
# THE CONVERGENCE RUN: A vs GDN vs B, 5 h each (~1.4B tokens, single pass on the XL
# corpus), so the gap's slope is read where the crossover would have to happen
# (0.6-1.1B tokens by extrapolation). One seed: the gap reproduced across seeds to
# 0.001 in the 537M long run, so a real 0.03-0.05 shows on the last 20 evals.
# Waits for the XL corpus AND for the lifted-cap run to finish. Eval every 600 s.
# NOTE: new val split -> absolute values are not comparable to the 537M runs, only
# the three arms among themselves.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
until grep -q "XL READY" runs/prep_xl.log 2>/dev/null && grep -q "LONG_FAST DONE" runs/long_fast.log 2>/dev/null; do sleep 60; done
LOG=runs/long5h.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval"
cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" $COMMON --save runs/ck_$label "$@"; }
cell l5_A_s0   --seed 0 --variant cshort_damph_cc   --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02
cell l5_gdn_s0 --seed 0 --variant gdn_cc            --ff 260
cell l5_B_s0   --seed 0 --variant cshort_damphkv_cc --Mc 190 --dv 56 --ff 440 --Ls 16 --theta-scale 0.02
echo "##### LONG5H DONE"
