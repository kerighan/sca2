#!/usr/bin/env bash
# Confirmation run for the two knobs that survived sweep_pycode.sh.
#
# What the sweep established, and what it did not:
#   * c_decay is the only knob that makes SCA2 use distance. Per-position slope
#     -0.159 against +0.009 for the baseline, ~60% of GDN's -0.276. Solid: it is
#     a within-model relative measure, and no other knob moves it at all.
#   * depth gives the steepest scaling slope of the sweep (-0.257 vs -0.059).
#   * the sweep's EXTRAPOLATIONS are not usable. `base` and `gc_off` are the same
#     function with knobs off; they agree to 0.004 nats at 60M tokens but their
#     fitted slopes differ by 0.074, which is 0.20 nats once extended to 176M.
#     So the ranking of gc_decay vs deep4 by projection was inside the noise.
#
# Hence: run to a FULL EPOCH and read the endpoint, not a fit. 3600s is a cap,
# not a budget -- every arm should hit "corpus exhausted" first (177.4M tokens at
# 54-98k tok/s = 1800-3300s), which makes this equal-tokens by construction.
#
# The 4-layer arms carry 2x the layer parameters of the 2-layer ones, so they are
# NOT matched against the existing GDN reference (2.889 nats, 2 layers, one epoch,
# runs/pycode.jsonl). That is why gdn4 is in here: without it, deep4 winning
# would only say that 743k parameters beat 371k.
#
# gdn4 needs ff=260, not the 364 the SCA2 arms use: GDN spends its parameters
# differently, and at ff=364 it would carry 849,768 layer params against 744,120
# for decay_deep4 -- a 14% head start. ff=260 puts it at 742,856, within 0.2%.
#
#   bash confirm_pycode.sh 2>&1 | tee runs/confirm_pycode.log
#   python dump_confirm.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/confirm_pycode.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128
        --Mc 128 --Md 4
        --seconds 3600 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"

cell () {                        # cell <label> <extra args...>
  local label=$1; shift
  echo "##### $label :: $*"
  python -u pretrain.py --label "$label" $COMMON "$@"
}

# decay alone, parameter-matched to the GDN reference already in runs/pycode.jsonl
cell decay2      --ff 364 --variant gc_cc          --layers 2 --c-decay
# depth alone, to separate it from the decay effect
cell deep4       --ff 364 --variant v3polarflat_cc --layers 4
# the combination the sweep never tested: the two independent levers together
cell decay_deep4 --ff 364 --variant gc_cc          --layers 4 --c-decay
# the reference that makes the 4-layer arms interpretable (see ff note above)
cell gdn4        --ff 260 --variant gdn_cc         --layers 4

echo "##### CONFIRM DONE"
