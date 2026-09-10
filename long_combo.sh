#!/usr/bin/env bash
# THE TWO FIXES TOGETHER (user's call: combination first, ablate after if notable):
#   long head  = cdelta_kv  : key-verification gate on the read  (CATCHUP.md)
#   D slot     = short dft C head, L=16                            (arch_short.py)
#   ff 364 -> 440 to match (186021 vs 185959). Longrun protocol, class breakdown.
# Then the single-change short arm, for attribution.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON --save runs/ck_$label "$@"; }
cell catch_shortkv_s0 --seed 0 --variant cshort_kv_cc --Mc 190 --dv 56 --ff 440 --Ls 16 --theta-scale 0.02
cell catch_short_s0   --seed 0 --variant cshort_cc    --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02
echo "##### COMBO QUEUE DONE"
