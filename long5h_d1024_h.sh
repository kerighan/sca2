#!/usr/bin/env bash
# d=1024: release the decay cap. This is the strongest finding of the campaign so far and
# it comes straight out of round 1's checkpoint, not from a guess.
#
# THE FINDING. The damped half of the long head's spectrum is FROZEN. lam is computed as
# softplus(lam_raw).clamp(max=lam_max) with lam_max = 1/L = 1/64, i.e. a floor of 64
# tokens on how fast a damped mode may forget. What the trained model asks for, reading
# softplus(lam_raw) BEFORE the clamp, is a memory of 23-34 tokens -- 1.5x to 2.4x past the
# cap, in 7 layers out of 8. And clamp() has ZERO GRADIENT above its bound, so every mode
# that goes over stops hearing the loss entirely and feels only weight decay. Half the
# spectrum (persist=0.5) sat disabled for the whole 5 h run with no gradient path out.
#
# It explains the three things that did not add up:
#   - raising M does nothing (the new modes get clamped the same way) -- M512 came in at
#     a median -0.018 nats over 19 points, inside the +-0.02-0.05 noise, for 1.7x the
#     compute and 3.6x the state;
#   - the gap is a CONSTANT offset at unchanged slope (0.63 sigma between the two slopes)
#     -- a fixed fraction of the spectrum is off from the first step to the last;
#   - only 57-183 of 256 modes carry read weight (participation ratio of |w|^2).
#
# WHERE IT CAME FROM. The rule lam_max = 1/L was introduced on 11 September so a damped
# mode never forgets faster than the short window remembers, and it was validated on COPY
# at the campaign's Ls=16 -- where it means a 16-token floor, and 23-34 is allowed. v1
# then moved the window to L=64 without revisiting it, which tightened the floor 4x and
# pushed it straight through the range the model wants. The other agent's rule that L, the
# damped modes' minimum memory and the rope base must stay coherent is exactly right; this
# is that rule being violated by the L=16 -> 64 change.
#
# --lam-max 0.0625 = 1/16 restores the campaign's effective floor, covers the asked-for
# range (0.024-0.044) with headroom, and leaves lam_max*chunk = 8 against a numerical
# limit near 60. mem_range stays at the aligned default (L, 32L) = (64, 2048): one change.
#
# Arm 2 adds the theta correction on top. theta is the weaker lead -- it is NOT clamped,
# so the model corrected it itself during round 1 (0.016 -> 0.058..0.317 by layer, against
# weight decay); a bad init there costs optimisation time, not necessarily final loss.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lapa_lam16      --theta-scale 0.02 --lam-max 0.0625
cell d1024_lapa_lam16_th02 --theta-scale 0.20 --lam-max 0.0625
echo "##### LONG5H_D1024_H DONE"
