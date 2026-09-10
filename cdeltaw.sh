#!/usr/bin/env bash
# Can the delta rule's gain be kept at the additive baseline's SPEED?
#
# cdelta_t02 closed the whole gap to GDN (n=3: 2.7480 vs md4_dv32 2.9058,
# t=-7.09; vs gdn4 2.7869, t=-0.90 i.e. parity) but costs 1.19x in throughput:
# 68,400 vs 81,400 tok/s, so SCA2 LOST the 12% speed edge it used to have over
# GDN and is now 6% slower than it.
#
# The FLOP accounting puts all of that cost in one term, the Gram matrix G, and G
# depends only on the WRITE phase. Freeze the content part of that phase and G
# becomes a constant Toeplitz matrix -- one per chunk size, cached forever --
# plus the C head stops reading h at all, removing a second matmul per layer.
# sca2/arch_cdelta.py::CHeadDeltaWPos, checked against the explicit Gram to
# 3.3e-16 and against the sequential recurrence to 2.3e-15.
#
# Parameter count is IDENTICAL to cdelta (744,044): theta keeps its M entries,
# it is simply no longer applied to h, and K stays fully used on the read. This
# is why the write phase is frozen and not all of theta -- freezing all of it
# would leave K unused, 193,536 dead parameters out of 743,528, and the arm
# would be 26% smaller in effect.
#
# WHAT IS BEING GIVEN UP, so the result is not oversold either way: the erase is
# now addressed by LAG instead of by content. It removes what was written at
# nearby positions per the grid's Dirichlet kernel, not the association at a
# matching key. Three outcomes, all informative:
#
#   keeps the gain   -> golden: GDN-parity quality at baseline speed, AND the
#                       mechanism was never content addressing in the first
#                       place -- it was whitening a near-degenerate code book
#                       (rope is rank 49/128 at T=128). Note cdelta_t0 did NOT
#                       test this, because its theta drifted from 0 to
#                       |theta| = 1.8..11.4; this arm cannot drift.
#   keeps part        -> the two effects are separable and additive; measure both.
#   loses it          -> content addressing IS the mechanism and the Gram has to
#                       be paid. Fall back to the CTX knob (cost is linear in
#                       chunk size, and chunking is provably exact) or to
#                       addressing the erase on a sub-band of the M frequencies.
#
# n=3 directly rather than a screen then a confirmation: the GPU is free and the
# effect to detect (does 0.158 survive) is large against a 0.015 seed sd, so the
# screen would not change the decision and would cost an extra 40 min.
#
#   setsid nohup bash cdeltaw.sh > runs/cdeltaw.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

while pgrep -f "pretrain.py --label cdelta" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG
        --dv 32 --Mc 378 --Md 4 --ff 364 --variant cdeltaw_cc
        --theta-scale 0.02"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell cdeltaw_t02_s0 --seed 0 --save runs/ck_cdeltaw_t02_s0
cell cdeltaw_t02_s1 --seed 1 --save runs/ck_cdeltaw_t02_s1
cell cdeltaw_t02_s2 --seed 2 --save runs/ck_cdeltaw_t02_s2

echo "##### CDELTAW DONE"
