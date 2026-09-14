#!/usr/bin/env bash
# d=1024: LayerScale -- a learned gain on each residual branch.
#
# Measured on the trained 10-layer checkpoint, which is what motivates it:
#   layer      0      2      5      8      9
#   ||x|| in  25.8  285.6  558.8  706.8  759.4     <- the stream grows 29x
#   ||dx||   202.7  142.5  190.1  134.5  216.2     <- each branch emits a CONSTANT norm
#   dx / x    7.85   0.50   0.34   0.19   0.29     <- so relative contribution collapses 41x
#   grad     1.1e-1 7.5e-2 7.3e-2 3.8e-2 5.0e-2    <- gradient decays only 3x
#
# The gradient does NOT vanish; AdamW normalises by RMS and 3x across ten layers is little.
# What collapses is the branch's share of the stream, because the LayerNorm at the head of
# each branch erases the stream's scale, so a branch cannot grow with depth without mix
# growing its own weights. That matches the per-layer loss ablation exactly: muting layer 0
# costs +1.590 nats, layer 8 costs +0.013.
#
# In principle this knob is redundant -- mix could scale itself. In practice giving the
# scale its own parameter and its own gradient, instead of leaving it entangled in a
# 4dv x d matrix, is what LayerScale does and it is known to help deep stacks. Two scalars
# per layer, initialised at 1, so it is bit-for-bit identical to the current layer at step
# one (verified, 0.00e+00) and decode == prefill holds with the gains randomised (1.3e-15).
#
# Read it as a discriminator: if it changes nothing, the deep layers genuinely have nothing
# to say and the depth result stands as a property of the model rather than of its
# parameterisation.
#
# --layers 10 on purpose: the effect it targets only appears with depth, and the 8-layer
# stack has already been measured without it. Read at MATCHED TOKENS against d1024_L10.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_decayin" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_decayin finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_L10_ls --layers 10 --layer-scale
echo "##### LONG5H_D1024_X DONE"
