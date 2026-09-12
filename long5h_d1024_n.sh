#!/usr/bin/env bash
# d=1024: the key-verification gate AND the causal conv, together.
#
# kv earned its place: cut short at 205M tokens it was running a median -0.036 nats under
# round 1 -- the first arm of the campaign to move at all, against M512's -0.018 (noise)
# and lam16's +0.002 (nothing). At matched tokens (~205M) it takes word_rep from 1.709 to
# 1.612 while word_new barely moves (5.220 -> 5.202): it helps, though not through the
# channel it was designed for. It closes about a quarter of the gap to GDN, so it is a
# component to keep rather than the answer on its own.
#
# The conv is the thing the layer never had. Mamba has one, GDN has one (kernel 4 on
# q/k/v, in the baseline we are behind), LFM2 is built from them. LayerCfg.conv existed
# but only arch_gatedc implemented it, so no cshort arm has ever run with a conv. It is now
# in ShortLayer, so every cshort variant inherits it -- 4.1k parameters per layer, identity
# at init (verified bit-for-bit against conv=0 in float64), decode == prefill to 1.1e-15.
#
# The checkpoint ablation motivates it independently: the SHORT head is this model's pillar
# (+6.45 nats when muted, against the long head's +2.77), so it leans hard on local
# structure -- which a conv supplies for 4k parameters instead of a memory's millions.
#
# Arm 2 then widens the pillar on top of the best base available.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant cshort_damphkv_cc --Mc 256 --dv 256 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_kv_conv4      --Ls 64  --conv 4
cell d1024_kv_conv4_L128 --Ls 128 --conv 4
echo "##### LONG5H_D1024_N DONE"
