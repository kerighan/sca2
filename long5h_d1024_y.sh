#!/usr/bin/env bash
# d=1024: LayerScale -- a learned gain on each residual branch. AT 8 LAYERS, not 10.
#
# The motivating pathology was first measured on the 10-layer checkpoint, but it is
# already present in the 8-layer stack every arm of this campaign has used. Measured on
# runs/ck_d1024_fast_kv_conv4 (our best arm, +0.046 vs GDN at equal wall clock):
#
#   layer      0      2      4      6      7
#   ||x|| in  24.7  255.4  414.8  596.1  658.7   <- the stream grows 27x
#   ||dx||   170.1  137.6  213.5  182.3  170.6   <- each branch emits a CONSTANT norm
#   dx / x    6.88   0.54   0.52   0.31   0.26   <- relative contribution collapses 26x
#   grad    1.11e-1 7.5e-2 6.6e-2 5.7e-2 5.0e-2  <- gradient decays only 2.2x
#
# So this is NOT a vanishing-gradient problem and NOT an artefact of adding depth. What
# collapses is the branch's share of the stream: the LayerNorm at the head of each branch
# erases the stream's scale, so a branch cannot grow with depth unless mix grows its own
# weights. It matches the per-layer loss ablation exactly (mute layer 0: +1.590 nats;
# mute layer 8 of the 10-stack: +0.013).
#
# Run at 8 layers rather than 10 for one reason: 8 is our best configuration, so a gain
# here is directly bankable against GDN, whereas a gain at 10 would be measured against a
# depth setting that has already been shown to buy nothing. The effect it targets is
# present at both depths (above), so depth is not needed to expose it.
#
# In principle the knob is redundant -- mix could scale itself. In practice giving the
# scale its own parameter and its own gradient, instead of leaving it entangled in a
# 4dv x d matrix, is what LayerScale does and it is known to help deep stacks. Two scalars
# per layer, initialised at 1, so it is bit-for-bit identical to the current layer at step
# one (verified, 0.00e+00) and decode == prefill holds with the gains randomised (1.3e-15).
#
# Read at MATCHED TOKENS against d1024_fast_kv_conv4: a single change, --layer-scale.
# Read it as a discriminator: if it changes nothing, the deep layers genuinely have
# nothing to say and the depth result stands as a property of the model rather than of
# its parameterisation.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_ls --layer-scale
echo "##### LONG5H_D1024_Y DONE"
