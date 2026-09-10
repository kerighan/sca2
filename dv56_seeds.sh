#!/usr/bin/env bash
# dv=56 / Mc=190, seeds 1 and 2. Two questions for the price of one.
#
# QUESTION 1 -- IS IT THE SPEED CHAMPION. dv56 screened quality-NEUTRAL against
# cdelta_t02 under the paired estimator (-0.002, seed 0) while running 77405
# tok/s, i.e. +13% on the dv32 champion against dv48's +9%. If neutral holds at
# n=3, the delta rule's throughput penalty drops from 1.19x to about 1.05x and
# dv56 becomes the shape to ship whenever speed is the binding constraint --
# dv48 keeps the quality crown at -0.021.
#
# QUESTION 2 -- IS THE PAIRED ESTIMATOR'S VARIANCE REAL, and this is the reason
# not to skip these runs. paired()'s between-seed sd on dv48 was ~0.005 against
# 0.043 at the endpoint, a factor of 8. Some of that is genuine (it estimates a
# window mean, not a point), but I have not verified it, and I have been reading
# n=1 screens with it. dv56 at n=3 measures it directly on a second arm:
#
#     paired sd ~ 0.005 again  -> the estimator is that tight, and the n=1 screen
#                                 readings above (dv40 +0.043, dv64 +0.055) can be
#                                 trusted as a shape curve.
#     paired sd ~ 0.04         -> its precision was an artefact of dv48's three
#                                 seeds happening to agree, every n=1 paired
#                                 number in this campaign is uninformative, and
#                                 the shape curve stays unestablished.
#
# Either answer is worth having. The first buys back the whole screening
# methodology; the second invalidates it, which is better learned now than after
# it has been quoted.
#
# WHY dv56 AND NOT dv40. dv40 is the arm whose sign FLIPPED between estimators
# (-0.046 endpoint, +0.043 paired), so it is the most tempting target -- but it is
# also slower than both dv48 and dv56 and screened at best neutral under either
# reading, so resolving it changes no decision. dv56 is on the speed frontier.
#
# 185959 params/layer against the champion's 186011 (-0.028%). State is 21280,
# BELOW both dv48 (24288) and dv32 (24192): if dv56 holds, the state-size reading
# of the shape effect is dead on both sides of the 2*Mc*dv peak at dv=40.
#
# ~1 h 20 total at 77.4k tok/s. Queues behind gdn_seeds.sh on the GPU wait.
#
#   setsid nohup bash dv56_seeds.sh > runs/dv56_seeds.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG
        --Md 4 --ff 364 --variant cdelta_cc --theta-scale 0.02 --dv 56 --Mc 190"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell shape_dv56_s1 --seed 1
cell shape_dv56_s2 --seed 2

echo "##### DV56 SEEDS DONE"
