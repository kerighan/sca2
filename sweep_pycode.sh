#!/usr/bin/env bash
# SCA2 configuration sweep on the long-document Python corpus.
#
# Motivation: on pycode_long1024 (whole-file windows, T=1024) GDN beats the
# baseline SCA2 by 0.29 nats at equal tokens, and the per-position profile says
# why -- GDN's loss falls -0.276 across the window, SCA2's only -0.030. So the
# question this sweep asks is: which knob makes SCA2 use distance?
#
# One factor at a time around the baseline, because 12 runs cannot resolve
# interactions. Two families, because the knobs live in different layer classes:
#   v3polarflat_cc -> SCA2Layer: Mc, Md, dv, ff, d, layers, freq, gated_read
#   gc_cc          -> GatedLayer: also c_decay, c_heads, c_sepq, conv
# Cell 9 (gc_cc, all knobs off) is the bridge: it should land on cell 1.
#
# Equal wall clock per cell, NOT equal steps, and 700s rather than a full epoch
# so the sweep fits in ~2.5h. Caveat that matters: against GDN the crossover in
# this corpus was at 859s, so a 700s ranking can inflate whichever arm is fast.
# Within the SCA2 family speeds are close, so the ranking is usable -- but the
# winner must be re-run to a full epoch before anything is concluded.
#
#   bash sweep_pycode.sh
#   python -c "..."   # see the analysis at the end of the run
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/sweep_pycode.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --layers 2 --d 128
        --seconds 800 --eval-batches 60 --eval-every 100 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"

cell () {                        # cell <label> <extra args...>
  local label=$1; shift
  echo "##### $label :: $*"
  python -u pretrain.py --label "$label" $COMMON "$@"
}

# --- family 1: v3polarflat_cc (the arm that lost) ------------------------- #
V="--variant v3polarflat_cc"
cell base        $V --Mc 128 --Md 4  --ff 364
cell Mc256       $V --Mc 256 --Md 4  --ff 364
cell Mc64        $V --Mc 64  --Md 4  --ff 364
cell Md16        $V --Mc 128 --Md 16 --ff 364
cell gatedread   $V --Mc 128 --Md 4  --ff 364 --gated-read
cell freqlen     $V --Mc 128 --Md 4  --ff 364 --freq len
cell wide192     $V --Mc 128 --Md 4  --ff 364 --d 192
cell deep4       $V --Mc 128 --Md 4  --ff 364 --layers 4

# --- family 2: gc_cc, the forgetting knobs ------------------------------- #
G="--variant gc_cc --Mc 128 --Md 4 --ff 364"
cell gc_off      $G
cell gc_decay    $G --c-decay
cell gc_conv4    $G --conv 4
cell gc_full     $G --c-decay --c-heads 4 --conv 4

echo "##### SWEEP DONE"
