#!/usr/bin/env bash
# d=1024, arms 3-4: correct theta_scale, then re-test M on top of the correction.
#
# WHY theta_scale. The round-1 checkpoint says the init is wrong, and says it directly
# rather than by inference. theta was initialised at 0.02*randn (mean|theta| 0.016) and
# came out of 5 h at mean|theta| 0.058 / 0.111 / 0.157 / 0.256 / 0.268 / 0.282 / 0.231 /
# 0.317 by layer -- a 4x to 20x rise, monotone with depth, AGAINST AdamW's default
# weight_decay of 0.01 pulling it toward zero. The gradient pushed on it for the whole
# run. theta sets how much CONTENT (theta.K(h)) there is against POSITION (s.omega) in
# the phase code, and 0.02 was settled at d=128 where ||K_m|| at init is 0.577 -- exactly
# what it is at d=1024, so the init never scaled with the width at all.
#
# It also fits the shape of the failure better than capacity does. The round-1 gap is a
# CONSTANT offset: slopes -0.3454 +- 0.0141 (LapA) vs -0.3321 +- 0.0155 (GDN), a 0.63
# sigma difference, i.e. indistinguishable. A capacity ceiling would flatten LapA's
# slope; it does not, and doubling the mixer (M and dv both, state 156k -> 566k) moved
# nothing. A mis-set constant costs a fixed amount everywhere at unchanged slope, which
# is the signature actually observed.
#
# The average learned mean|theta| is 0.193, which an init would reach at theta_scale
# 0.242. 0.20 is the arm: above every layer's starting point, still below what the deep
# layers converged to, and no single global value can suit both ends of that gradient.
#
# Costs nothing in parameters, state, or throughput, so it is readable at matched tokens
# AND at equal wall clock -- unlike the ff-matched arm it replaced, which would have
# traded one for the other.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_lapa_M512" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_lapa_M512 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --dv 256 --Ls 64 --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

# single change against round 1 (d1024_lapa): theta_scale 0.02 -> 0.20
cell d1024_lapa_th02   --Mc 256  --theta-scale 0.20
# single change against the arm above: M 256 -> 1024, theta kept corrected
cell d1024_lapa_th02_M1024 --Mc 1024 --theta-scale 0.20
echo "##### LONG5H_D1024_F DONE"
