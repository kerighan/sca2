#!/usr/bin/env bash
# d=1024: LayerScale + FREE MODES, launched directly (d1024_ls was cut at 289.5M).
# Full rationale in long5h_d1024_z.sh; the short version:
#
#   The 8 trained layers' 16 temporal profiles span an EFFECTIVE RANK OF 2.82 (97.7% of
#   the singular mass on 2) because softplus(a).clamp(max=lam_max) plus the `persist` pin
#   makes the reachable lambda a TWO-POINT set, and 41-100% of each layer's free modes sit
#   exactly at the clamp where the gradient is zero and no mode ever escapes (layer 5:
#   128/128). GDN, measured the same way, spans 2.5 .. 5.8e6 tokens with a x619 depth
#   gradient in its median memory. The link function is fine -- for lam <~ 1/64 softplus
#   IS exp, |d log tau / da| = 0.94..1.00 -- the BOUND is the problem.
#
# --lam-free: lambda = exp(a), nothing pinned, ceiling at 55/chunk = 0.43 (memory floor
# 2.3 tokens). --damp-mem 4,20000 spreads the init log-uniformly over that whole range.
#
# --layer-scale carried over: d1024_ls read -0.027 vs base (median, negative on 8/8 points
# above 180M, sd 0.016) before being cut. A small real gain, kept.
#
# READING RULE, decided after today: our resolution is ~0.03 (two arms computing the SAME
# function -- kv_conv4 vs fast_kv_conv4 -- differ by 0.028 at equal wall clock and by
# 0.065 sd in the 100-200M window). Do NOT read this arm below 200M tokens, and do not
# call anything under 0.05 a win.
#
# NOTHING ELSE RUNS ON THE GPU WHILE THIS ARM IS UP. d1024_ls's throughput was depressed
# by analyses I ran concurrently during its first two hours; its equal-wall-clock number
# is pessimistic by an unknown amount and that must not happen again.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lsfree --layer-scale --lam-free --damp-mem 4,20000
echo "##### LONG5H_D1024_Z2 DONE"
