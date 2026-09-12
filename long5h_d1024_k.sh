#!/usr/bin/env bash
# d=1024: stop repairing what is dead, fund what carries the loss.
#
# The ablation on round 1's trained checkpoint (eval only, no retraining) says which
# mechanisms are load-bearing and which are not:
#
#   short head muted        +6.45 nats   <- the pillar
#   long head muted         +2.77
#   theta = 0               +2.10
#   PERSISTENT modes muted  +2.04
#   beta -> 0 (delta rule)  +1.42
#   slow modes muted        +0.52
#   fast modes muted        +0.47
#   DAMPED modes muted      +0.13        <- half the spectrum, near-dead
#
# Muting 128 of 256 modes costs 0.13 nats while the persistent half costs 2.04 -- a factor
# of 15 between the two halves of the same head. That retro-explains the whole day: we
# spent it releasing the decay cap on modes that do not carry the loss, so --lam-max
# 0.0625 landing at +0.002 was predictable from this measurement. It also matches the
# mechanism the other agent described: the damped modes converge to the cap, i.e. a memory
# of 64 tokens = exactly the short head's window, and duplicate its job. Here we can see
# which of the two overlapping mechanisms wins, and it is the short head by 2.3x.
#
# ARM 1  --persist 0.875   Reallocate the spectrum toward what works: 88% persistent
#                          instead of 50%. Zero extra parameters, zero extra state.
#
# ARM 2  --Ls 128          Widen the pillar. The short head is the most load-bearing thing
#                          in the layer and its window is set to 64 by convention, not by
#                          measurement. +0.065M params/layer, state 156k -> 197k. NOTE this
#                          is not a pure single change: lam_max and mem_range default to
#                          1/L and (L, 32L), so they follow the window -- which is the
#                          intended coupling, and in any case only touches the damped modes
#                          that the ablation just showed are worth 0.13 nats.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_kv" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_kv finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_persist875 --Ls 64  --persist 0.875
cell d1024_L128       --Ls 128
echo "##### LONG5H_D1024_K DONE"
