#!/usr/bin/env bash
# Promote the shape screen winner, and bracket the optimum from above.
#
# WHAT sweep_shape.sh FOUND (n=1, seed-paired against cdelta_t02_s0 = 2.7447):
#
#     dv   Mc     val      delta     in sd    tok/s
#     24  441   2.8810   +0.1363     +9.3    65094   <- falsification arm, lost
#     32  378   2.7447        --       --    68360   <- champion
#     40  315   2.7262   -0.0185     -1.3    69486
#     48  253   2.6773   -0.0674     -4.6    74564   <- promote
#     64  128   2.7837   +0.0390     +2.7    84370
#
# dv=48 cleared the 0.022 promotion bar by 3x AND ran 9% faster, cutting the delta
# rule's throughput penalty from 1.19x to 1.09x. Both halves of the bet landed.
#
# ALLOCATION, NOT CAPACITY -- and this was settled for free. I had planned dv=24 vs
# dv=56 as the disambiguating pair (equal state, opposite ends). But dv=32 and
# dv=48 do it better: their C states are 24192 and 24288, 0.4% apart, essentially
# identical -- and dv=48 wins by 0.067. So the gain is not extra state. The
# counterexample seals it: dv=64 holds 16384, a full 23% LESS state than dv=24,
# yet scores 0.097 BETTER. val is not monotone in state size here.
#
# TWO THINGS TO SETTLE, in this order:
#
# 1. PROMOTION (arms 1-2). dv=48 seeds 1 and 2, so it is 3-vs-3 against
#    cdelta_t02 (2.7480, sd 0.0147) and against gdn4. GDN is the reason this
#    cannot be called yet: its seed sd is 0.0733, the worst in the campaign, so
#    the apparent 0.11 margin over GDN's mean is inside the noise at n=1. The
#    paired estimator in dump_cdelta.py is what decides.
#
# 2. BRACKET (arm 3). dv=48 wins, dv=64 loses, so the interior optimum sits in
#    [40, 64]. dv=56 / Mc=190 asks whether it lies beyond 48. Note its state is
#    21280, LOWER than dv=48's -- past the dv=40 peak of the quadratic 2*Mc*dv --
#    so if dv=56 also beats the champion, the state-size reading dies a second
#    death, on the other side of the peak.
#
# Params: 186020 (+0.005%) and 185959 (-0.028%) against the champion's 186011,
# solved against the real builder. ~2 h; dv=56 is the fast one at Mc=190.
#
#   setsid nohup bash shape_confirm.sh > runs/shape_confirm.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG
        --Md 4 --ff 364 --variant cdelta_cc --theta-scale 0.02"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell shape_dv48_s1 --dv 48 --Mc 253 --seed 1
cell shape_dv48_s2 --dv 48 --Mc 253 --seed 2
cell shape_dv56_s0 --dv 56 --Mc 190 --seed 0

echo "##### SHAPE CONFIRM DONE"
