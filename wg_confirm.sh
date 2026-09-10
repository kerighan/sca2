#!/usr/bin/env bash
# WG=2, full epoch, n=3 -- the only protocol that can call this.
#
# Why this arm. The 900s probes could NOT resolve absolute val: mid-descent, wg2
# minus wg1 at matched tokens swung +0.055, -0.181, -0.183, -0.085, -0.087,
# +0.024 across six evals. Both curves are dropping steeply there, so a small
# horizontal offset in when a drop lands shows up as a huge vertical difference.
# pretrain.py::evaluate already says so: the eval set is fixed and sequential, so
# this is trajectory noise, not measurement noise, and more eval batches cannot
# fix it.
#
# What WAS stable is the WITHIN-MODEL position profile, which the same docstring
# recommends for exactly this reason. slope = pos[7] - pos[0], wg2 minus wg1, at
# five matched token counts:
#       -0.077   -0.105   -0.097   -0.125   -0.078
# Consistent in sign and size. The axis is peaked at WG=2 (wg4 about -0.03,
# wg8 nil), and WG=2 costs 1.13x throughput against 1.63x for WG=4.
#
# So: WG=2 shifts roughly 0.05 of capability from short positions to long ones.
# Whether that is a NET win on val is what this run settles.
#
# Control is md4_dv32 at n=3, already logged (runs/md_axis.jsonl seed 0,
# runs/seeds.jsonl seeds 1-2) under exactly these flags -- not rerun.
#
# --seconds 3600 exhausts the 173M-token corpus first (wg2 at 70.7k tok/s needs
# 2447s), so every arm is one full epoch, same as the control.
#
#   setsid nohup bash wg_confirm.sh > runs/wg_confirm.log 2>&1 < /dev/null &
#   python dump_wg_confirm.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/wg_confirm.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 3600 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"

# ff 361 matches the control's 743,528 layer params to within -60.
ARM="--dv 32 --Mc 378 --Md 4 --ff 361 --variant wg2_cc"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell wg2_s0 --seed 0 $ARM
cell wg2_s1 --seed 1 $ARM
cell wg2_s2 --seed 2 $ARM

echo "##### WG CONFIRM DONE"
