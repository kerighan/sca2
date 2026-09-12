#!/usr/bin/env bash
# d=1024, arm 2: everything the round-1 checkpoint complained about, released at once.
#
# Arm 1 (d1024_lapa_lam16, running) is the clean single change: --lam-max 0.0625 alone,
# so the decay-cap finding gets an unambiguous reading whatever this arm does.
#
# This arm adds the two remaining complaints from the same checkpoint:
#
#   --learn-persist   lambda = lam_max*sigmoid(a) instead of softplus(a).clamp(max=lam_max)
#                     with half the modes hard-pinned at zero. Two problems in one: the
#                     clamp has ZERO GRADIENT above its bound (80-99% of damped modes sat
#                     there all run, deaf to the loss), and pinning exactly 50% of the
#                     spectrum persistent is a guess, not a measurement. Here nothing is
#                     pinned, --persist only sets where the split STARTS, and the gradient
#                     decides it. At init: lambda from 2e-5 (memory ~48k tokens, infinite
#                     for a 1024 window) up to the cap, continuously.
#
#   --theta-scale 0.20  theta came out of round 1 at mean|theta| 0.058..0.317 by layer
#                     from an init of 0.016 -- 4x to 20x, monotone with depth, against
#                     weight decay. The average learned value corresponds to an init of
#                     0.242. This is the WEAKER of the leads, deliberately bundled here
#                     rather than given its own arm: theta is not clamped, so the model
#                     already corrects it itself; a bad init costs optimisation time, not
#                     necessarily final loss.
#
# Attribution if it wins: arm 1 gives lam_max alone, the difference gives (persist+theta)
# jointly, and those two get separated afterwards. Bundling is a deliberate trade of
# attribution for wall clock, not an oversight.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_lapa_lam16" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_lapa_lam16 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lapa_free --lam-max 0.0625 --learn-persist --theta-scale 0.20
echo "##### LONG5H_D1024_I DONE"
