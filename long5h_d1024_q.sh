#!/usr/bin/env bash
# d=1024, round 3: the maximalist arm -- every mechanism on, and parameter-matched to GDN.
#
# Deliberately NOT a single change. The point is to find the ceiling of this direction
# first and subtract afterwards; each part is separately verified and separately
# revertible, and each is a strict generalisation that starts from the previous function.
#
#   --kv-dk 16       key verification. The gate reads the RETRIEVAL, not the input: it
#                    stores the write key beside the value and gates on cos(key read back,
#                    Kv(z)). Best mechanism at d=128, and at d=1024 it took a median -0.036
#                    nats off round 1 on its own. Ported into lapa/layer.py, exact against
#                    sca2's cshort_damphkv at 8.88e-16 on both prefill paths and decode.
#   --kv-gate-pc     that gate PER CHANNEL: ga, gb as 2*dv vectors instead of scalars, so
#                    each output channel gets its own slope and bias on the same evidence.
#                    Both reference architectures gate per channel; ours applied one number
#                    to 2*dv. Identical at init (8.88e-16 with the scalars broadcast in).
#   --conv 4         causal depthwise conv before both heads. Mamba, GDN and LFM2 all have
#                    one; this lineage never did. 4.1k parameters, identity at init.
#   --beta-groups 3  the erase gate per spectral BAND (slow integrators / persistent-fast /
#                    damped: 64/64/128 modes). The spectrum is deliberately heterogeneous
#                    and was being written with ONE strength; a fast pole holds short-lived
#                    content to overwrite hard, a persistent pole holds document memory that
#                    should not be. Folds into the query-side codes, so the chunked closed
#                    form is untouched. Identical at init, decode == prefill at 1.3e-15 with
#                    a randomised gate.
#   --ff 5729        parameter-matched to GDN: 13.6712M/layer against 13.6696M, +0.011%,
#                    142.9M model on both sides. Removes the last confound round 1 could not
#                    answer -- LapA has been the smaller model in every arm so far.
#
# All four mechanisms together cost 23.6k parameters per layer over plain v1. The ff is
# what closes the remaining 3.3M, which is the honest way to match: it funds the FFN, the
# part both architectures share, rather than inflating the mixer under test.
#
# COST, stated up front: ff 4096 -> 5729 is ~+40% on the FFN, roughly half the layer, so
# expect ~10% fewer tokens in 5 h. That helps the matched-token reading and hurts the
# equal-wall-clock one. Both will be reported.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_kv_conv4" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_kv_conv4 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_max      --ff 5729 --kv-dk 16 --kv-gate-pc --conv 4 --beta-groups 3
# arm 2 STACKS rather than ablates. --persist 0.875 hands 88% of the spectrum to the
# infinite-memory half instead of 50%, which the checkpoint ablation asks for directly:
# muting the persistent modes costs +2.04 nats, muting the damped half costs +0.13. The
# damped modes converge to the cap, i.e. a memory of exactly the short head's window, and
# duplicate a mechanism that beats them 2.3x -- so the budget is better spent elsewhere.
# Free: no parameters, no state, no throughput. Single change against d1024_max.
cell d1024_max_p875 --ff 5729 --kv-dk 16 --kv-gate-pc --conv 4 --beta-groups 3 --persist 0.875
echo "##### LONG5H_D1024_Q DONE"
