#!/usr/bin/env bash
# Three more GDN seeds. The comparison that matters is limited by GDN's noise,
# not by ours.
#
# WHY GDN AND NOT MORE SCA2 SEEDS. Welch's denominator for cdelta vs gdn4 at n=3:
#
#     se^2 = 0.0147^2/3 + 0.0733^2/3 = 7.2e-5 + 1.79e-3
#
# GDN supplies 96% of the variance. So a GDN seed buys ~25x more resolution per
# hour of GPU than an SCA2 seed does, and the instinct to shore up our own arm
# would have been close to useless. gdn4 seeds 3, 4, 5.
#
# ONE SEED CARRIES THE WHOLE STORY, which is the real reason this is needed.
# gdn4 = [2.8386, 2.7030, 2.8191]: s1 sits 0.12 below its two sisters and is the
# only reason parity with GDN is unresolved. It is also why the two estimators
# disagree by a factor of two -- endpoint interpolation reads
# cdelta_t02 - gdn4 = -0.0389, the paired multi-point reads -0.0832. Both are
# dominated by s1. I do not know which to believe, and that is itself the symptom
# of n=3 being too small on this arm.
#
# WHAT THIS CAN AND CANNOT SETTLE, stated before the fact so it cannot be
# rationalised after. At n=6 GDN vs n=3 SCA2, se = 0.0311, dof ~ 6:
#
#     if the true gap is -0.083  ->  t = 2.67 > 2.45, RESOLVED in our favour
#     if the true gap is -0.039  ->  t = 1.25,        STILL UNRESOLVED
#
# So this run only decides the question if the paired estimator was the right
# reading. If it lands in between, the honest outcome is "parity, not superiority"
# and the campaign should stop claiming a win over GDN rather than buy more seeds.
#
# MATCHING, unchanged and slightly against us: gdn4 uses --ff 260 to land at
# 185714 params/layer against the champion's 186011, i.e. GDN runs with 0.16%
# FEWER parameters. Command line is copied verbatim from seeds.sh so seeds 3-5 are
# drawn from the same arm as 0-2; only --seconds differs (4200 vs 3600), which is
# slack, not extra training -- both finish the single epoch and stop on corpus
# exhaustion at ~177.4M tokens.
#
# ~2 h at GDN's 72.3k tok/s. Runs after shape_confirm.sh via the GPU wait below.
#
#   setsid nohup bash gdn_seeds.sh > runs/gdn_seeds.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"
GDNARM="--ff 260 --variant gdn_cc"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell gdn4_s3 --seed 3 $GDNARM
cell gdn4_s4 --seed 4 $GDNARM
cell gdn4_s5 --seed 5 $GDNARM

echo "##### GDN SEEDS DONE"
