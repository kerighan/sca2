#!/usr/bin/env bash
# The missing control: the ADDITIVE write at theta init 0.02.
#
# Why it is missing, and why that is a real gap. The champion comparison I have
# been quoting is
#     cdelta_t02 (theta init 0.02)  -  md4_dv32 (theta init 0)  =  -0.1577
# and pretrain.py's --theta-scale defaults to 0.0, which neither md_axis.sh nor
# seeds.sh overrides. So md4_dv32 ran at theta init 0 and that -0.1577 confounds
# TWO changes: the delta rule and the theta init. It must not be quoted as "the
# delta rule's effect", which is what best_layer.py, WINNERS.md and the sca2
# README all did until now.
#
# What is already clean, at MATCHED init 0:
#     cdelta_t0 - md4_dv32 = -0.1179   Welch t=-5.01
# so the mechanism carries the large majority of the gain on its own. The
# remaining -0.0398 (t=-2.80) is the init, and it is only known to help IN THE
# PRESENCE of the delta rule -- under the additive write, theta 0.02 measured
# +0.012 WORSE at n=1 (best_layer.py FACTS). This run gives that comparison three
# seeds so the decomposition closes:
#
#     md4_t02 - md4_dv32          the init's effect under the ADDITIVE write
#     cdelta_t02 - md4_t02        the mechanism's effect at matched init 0.02
#
# Two outcomes, both worth having:
#   md4_t02 ~ md4_dv32  -> the init is inert without the delta rule, so the two
#                          effects are not additive and the init is doing
#                          something specific to error correction (plausible: it
#                          decorrelates the write codes, which is exactly what
#                          the Gram matrix in the solve conditions on).
#   md4_t02 < md4_dv32  -> part of the -0.158 was never about the delta rule and
#                          generation 1 was simply under-tuned. The mechanism's
#                          credit shrinks to -0.118 and stays there.
#
# Cheap: this is the baseline arm, 81.4k tok/s, ~2100 s/epoch, so ~1.8 h total.
#
#   setsid nohup bash theta_ctrl.sh > runs/theta_ctrl.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

while pgrep -f "pretrain.py --label cdeltaw" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG
        --dv 32 --Mc 378 --Md 4 --ff 364 --variant v3polarflat_cc
        --theta-scale 0.02"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell md4_t02_s0 --seed 0
cell md4_t02_s1 --seed 1
cell md4_t02_s2 --seed 2

echo "##### THETA CTRL DONE"
