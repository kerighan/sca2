#!/usr/bin/env bash
# The shape retrade, at MATCHED params: trade C-head modes (Mc) for value width
# (dv), under the error-correcting write.
#
# WHY NOW. The shape optimum dv=32, Mc=378, Md=4 was found by md_axis.sh under the
# ADDITIVE write -- a memory that accumulated its own read-out interference. More
# modes was then a way to keep distinct writes from colliding. The delta rule
# CORRECTS that interference instead, so the marginal value of a mode should have
# fallen, and some of the 378 may be better spent on richer values. This is the
# only untested knob that could plausibly move the champion, and it is doubly
# attractive because the Gram cost is linear in Mc: if fewer modes suffice, the
# quality retrade and the 1.19x throughput penalty pay for each other.
#
# THE BUDGET. Solved against the real builder, not by hand (Mc costs 131
# params/mode -- K plus theta, wr, wi; dv costs 1024/unit because it crosses C.V,
# D.V, D.qb_r, D.qb_i and mix(4*dv,d)):
#
#     dv    Mc   params/layer   vs champion   C state = 2*Mc*dv
#     24   441       186072        +0.033%         21168
#     32   378       186011         0.000%         24192   <- champion
#     40   315       185950        -0.033%         25200
#     48   253       186020        +0.005%         24288
#     56   190       185959        -0.028%         21280
#     64   128       186029        +0.010%         16384
#
# All inside the 0.07% the campaign has been matching to. NOTE, and this must not
# be glossed: at fixed params Mc(dv) is linear, so the state size 2*Mc*dv is
# QUADRATIC in dv and peaks near dv=40. This sweep therefore moves state size as
# well as its shape -- from 16.4k at dv=64 to 25.2k at dv=40. If the result tracks
# state size rather than dv, that is a different conclusion (capacity, not
# allocation) and the arms above are ordered to make the two separable: dv=24 and
# dv=56 have nearly equal state (21.2k, 21.3k) at opposite ends of the axis.
#
# Four arms, dv = 24 / 40 / 48 / 64, seed 0, one full epoch each. dv=24 runs FIRST
# on purpose: it is the falsification arm. If moving AWAY from values and towards
# modes wins, the premise is simply wrong and the rest of the sweep is a different
# experiment than the one described above.
#
# n=1 SCREENS ONLY. cdelta_t02's seed sd is 0.0147, so the promotion rule from
# WINNERS.md applies: an arm earns three seeds by beating its control by more than
# ~1.5 sd, i.e. ~0.022, on a full epoch. The control is cdelta_t02_s0 = 2.7447,
# already in runs/cdelta.jsonl at the same seed, so every comparison here is
# seed-PAIRED -- which is the sensitive one, and the only kind worth running at
# n=1.
#
# Speed is read as a second axis, not a tiebreak: lower Mc must come out faster
# (the Gram and the K2 kernel are both O(C^2.Mc)), and if a low-Mc arm holds
# quality then the 1.19x is refunded at the same time.
#
# ~2.6 h total, the low-Mc arms being the fast ones.
#
#   setsid nohup bash sweep_shape.sh > runs/sweep_shape.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

# Do not contend with anything already on the GPU.
while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG
        --Md 4 --ff 364 --variant cdelta_cc --theta-scale 0.02 --seed 0"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell shape_dv24_s0 --dv 24 --Mc 441
cell shape_dv40_s0 --dv 40 --Mc 315
cell shape_dv48_s0 --dv 48 --Mc 253
cell shape_dv64_s0 --dv 64 --Mc 128

echo "##### SWEEP SHAPE DONE"
