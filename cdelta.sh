#!/usr/bin/env bash
# Does a complex error-correcting write close the gap to GDN?
#
# Every capacity-reshaping lever is now exhausted and none of them moved the
# 0.08 nats: Mc/Md/dv (md_axis.sh, seeds.sh), depth, gated read-out, c_decay,
# conv, theta (theta.sh: 0 vs 0.02 unresolved) and per-value-group spectral
# weights (wg_confirm.sh: a NULL at a full epoch, so the rank-2 limit on the
# temporal profile was not the binding constraint). What is left is the one
# structural difference: GDN's write can take back what it stored at a key,
# SCA2's additive write cannot. sca2/arch_cdelta.py adds exactly that and
# nothing else -- see its docstring for the derivation.
#
# beta = 0 reproduces the baseline BIT-FOR-BIT (checked, 1.1e-15), so this arm
# is a strict extension of md4_dv32 and the gate can switch itself off. Cost is
# +516 params out of 743,528 (0.07%, the beta projection), so ff stays at 364
# and the arm is otherwise IDENTICAL to SCA2ARM in seeds.sh.
#
# TWO ARMS, and they test different mechanisms because the erasure's selectivity
# depends on theta (arch_cdelta docstring, "WHAT SELECTIVITY"):
#
#   t0    theta = 0. The code is purely positional, so G[t,s] = kappa(t-s) is
#         the Gram matrix of the rope dictionary -- which is documented as
#         near-degenerate (rank 49/128 at T=128, ref.freq_grid) -- and the solve
#         is its Gram-Schmidt whitening. Tests redundancy of the code book, not
#         erasure. Note this arm is a no-op by construction on freq=dft, where
#         kappa is a clean delta; it only means something on rope.
#   t02   theta = 0.02. The code depends on content, so the erase removes the
#         association at a CONTENT key. This is the actual GDN mechanism. 0.02
#         is used because theta.sh already showed it is not harmful on its own,
#         which keeps this arm's difference attributable to the delta rule.
#
# HONESTY ABOUT POWER. n=1 per arm at a full epoch. The seed sd on val is 0.050
# for md4_dv32 (n=3, dump_seeds.py), so n=1 can only SCREEN: it can reject an
# arm that fails to move, and it cannot confirm one that does. Anything that
# lands below the baseline mean by more than ~0.075 earns three seeds; anything
# inside that band is "not resolved" and gets dropped, exactly as wg2 was.
# The wg2 lesson is the reason both arms run a FULL epoch and are read on val
# only: mid-descent probes and the position slope both misled once already.
#
# Controls are NOT rerun -- md4_dv32 and gdn4 at n=3 are already in
# runs/md_axis.jsonl, runs/seeds.jsonl and runs/confirm_pycode.jsonl.
#
#   setsid nohup bash cdelta.sh > runs/cdelta.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

# Wait out any run still holding the GPU: two arms sharing it would corrupt the
# tok/s of both, and the clock-based eval schedule turns that into a shorter
# epoch rather than just a slower one.
while pgrep -f "pretrain.py --label wg2" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG --seed 0
        --dv 32 --Mc 378 --Md 4 --ff 364"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

# --save is not optional here: if beta stays at its 0.12 init the arm IS the
# baseline and a null says nothing about erasure. dump_cdelta.py reads it back.
cell cdelta_t0   --variant cdelta_cc --theta-scale 0.0  --save runs/ck_cdelta_t0
cell cdelta_t02  --variant cdelta_cc --theta-scale 0.02 --save runs/ck_cdelta_t02

echo "##### CDELTA DONE"
