#!/usr/bin/env bash
# Overnight long-horizon run: does generation 3's advantage over GDN SURVIVE more
# tokens, or does it close?
#
# THE QUESTION IS NOT THE ONE IT LOOKS LIKE. The intuition "if there is a real
# advantage the gap should widen" is not what the existing data shows. Fitting the
# gap against log-tokens over the campaign's own 40M..170M window, n=3 vs n=6:
#
#     tokens    gen3     gdn      gap
#      40.0M  3.6472  3.9934  -0.3463
#      68.8M  3.1667  3.5213  -0.3547
#      97.7M  2.9606  3.1531  -0.1925
#     126.5M  2.8675  2.9819  -0.1144
#     169.8M  2.7088  2.7921  -0.0833
#
#     d(gap)/d(ln tokens) = +0.222   95% CI [+0.163, +0.277]  (seed bootstrap)
#     dval/dln(tokens):  gen3 -0.624   gdn -0.846
#
# The gap is CLOSING, monotonically, and the CI excludes zero. GDN is descending
# faster. Naively extended the gap reaches zero near 340M tokens -- inside reach
# of this run. So the honest framing is that generation 3 may be winning only a
# TRANSIENT: better early, overtaken later. That is the hypothesis this run tests,
# and the outcome that would retract the headline.
#
# Why the slope is trustworthy at these seed counts even though the endpoint is
# marginal: a seed changes a run's LEVEL far more than its SHAPE, so the gap's
# slope against log-tokens largely cancels the seed offset. That is also why n=2
# below is defensible -- it is the trend, not the endpoint, that this run buys.
# DO NOT read the final val here as a new headline number: at n=2 it cannot
# resolve 0.08 nats.
#
# WHAT IT CANNOT SETTLE. The learning rate is CONSTANT in pretrain.py (no
# schedule), so neither arm is converged at any point on this curve and "final
# loss" means "loss at the budget", not "converged loss". A closing gap under
# constant LR could also be a decayed-LR artifact waiting to happen. The separate
# cheap experiment is a cosine schedule at the ORIGINAL 177M budget; it is not in
# here because mixing the two axes in one night settles neither.
#
# BUDGET, from a smoke test on this exact corpus rather than from the steady-state
# bench: pretrain.py's cumulative tok/s runs ~69k for both arms once the clock ramp
# is included (the 76.2k/70.7k in sweep_chunk.py are steady-state windows, which a
# whole-run average never reaches). 536.6M tokens at ~69k = ~2.16 h/run, so
#   4 runs ~= 8.6 h, and a 22:00 start lands ~06:40.
#
# --seconds is a CAP, not a budget: every arm should print "corpus exhausted"
# first, which makes the comparison equal-tokens by construction. The cap is set to
# 9000 s, only ~15% above the expected 7800 s, deliberately: if the machine slows
# down overnight the cost is a slightly short run, not a 12-hour overrun that is
# still going at breakfast. If an arm DOES hit the cap the arms end at different
# token counts -- dump_longrun.py interpolates to a common grid, so it degrades
# gracefully, but the endpoint comparison is then void.
#
# Arms alternate so that a machine change overnight cannot align with one arm.
#
# Smoke-tested: both arms start, compile and eval on this corpus, at 743836 vs
# 742856 layer params (0.13% apart, in GDN's favour). Do NOT read the tok/s in the
# log as a speed result -- it is a cumulative average and these runs are
# sequential, which is exactly the unblocked comparison sweep_chunk.py invalidated.
#
#   bash prep_longrun.sh           # FIRST, and it must finish before this runs
#   setsid nohup bash longrun.sh > runs/longrun.log 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16   # measured optimum, see sweep_chunk.py

DATA=pycode_long1024_big.pt
if [ ! -f "$DATA" ]; then
  echo "FATAL: $DATA missing -- run prep_longrun.sh first"; exit 1
fi

LOG=runs/longrun.jsonl
COMMON="--data $DATA --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG"

# Generation 3's champion shape, and GDN at ff=260 to match parameters (185959
# vs 185714 per layer). At a common ff=364 GDN would carry 14% MORE.
GEN3="--variant cdelta_cc --Mc 190 --dv 56 --ff 364 --theta-scale 0.02"
GDN="--variant gdn_cc --ff 260"

cell () { local label=$1; shift; echo "##### $label :: $*"; \
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell long_gen3_s0 --seed 0 $GEN3
cell long_gdn_s0  --seed 0 $GDN
cell long_gen3_s1 --seed 1 $GEN3
cell long_gdn_s1  --seed 1 $GDN

echo "##### LONGRUN DONE"
echo "read it with: python dump_longrun.py"
