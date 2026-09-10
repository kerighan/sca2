#!/usr/bin/env bash
# Two questions left open by md_axis.sh, in one overnight run.
#
# 1. Mc or C-state? Along the equal-parameter family at Md=4, ff=364, the C-head
#    state is 2*Mc*dv and it PEAKS at dv=32:
#        dv 64 -> Mc 128 -> state 16384   (deep4,     2.9188)
#        dv 32 -> Mc 378 -> state 24192   (md4_dv32,  2.8760)
#        dv 16 -> Mc 504 -> state 16128   (this run)
#        dv  8 -> Mc 566 -> state  9056
#    So dv cannot buy more state than md4_dv32 already has. But dv=16 has MORE
#    Mc than md4_dv32 and the SAME state as deep4, which separates the two
#    candidate explanations: land near 2.919 and the operative variable is state,
#    beat 2.876 and it is Mc (addressing capacity) instead.
#
# 2. Is the remaining 0.037 real? md4_dv32 (2.8760) vs gdn4 (2.8394) sits at
#    about 1 sigma: the residual sd of val around its local trend over the last
#    6 evals is 0.013 for md4_dv32 but 0.068 for gdn4. Two more seeds each gives
#    n=3 per arm, which is the minimum that can call it.
#
# Only the init varies with --seed; batches() is sequential, so every seed sees
# identical data in identical order.
#
#   setsid nohup bash seeds.sh > runs/seeds.log 2>&1 < /dev/null &
#   python dump_seeds.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/seeds.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 3600 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"

SCA2ARM="--dv 32 --Mc 378 --Md 4 --ff 364 --variant v3polarflat_cc"
GDNARM="--ff 260 --variant gdn_cc"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

# question 1: one arm, seed 0, comparable to the md_axis arms
cell md4_dv16   --seed 0 --dv 16 --Mc 504 --Md 4 --ff 364 --variant v3polarflat_cc

# question 2: seeds 1 and 2 (seed 0 already in runs/md_axis.jsonl and
# runs/confirm_pycode.jsonl respectively)
cell md4_dv32_s1 --seed 1 $SCA2ARM
cell gdn4_s1     --seed 1 $GDNARM
cell md4_dv32_s2 --seed 2 $SCA2ARM
cell gdn4_s2     --seed 2 $GDNARM

echo "##### SEEDS DONE"
