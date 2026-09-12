#!/usr/bin/env bash
# d=1024: the causal conv the layer never had.
#
# Every competitive linear mixer puts a short causal depthwise conv in front of its state:
# Mamba does, GDN does (kernel 4 on q/k/v -- it is in the baseline we are losing to), LFM2
# is built out of them. lapa/layer.py had none. sca2 had the knob (LayerCfg.conv) but only
# arch_gatedc ever implemented it, so no cshort/lapa arm has ever run with one.
#
# The ablation supports it: the SHORT head is the pillar of this model (+6.45 nats when
# muted, against the long head's +2.77), i.e. the model leans hard on local structure --
# exactly what a conv supplies, and far more cheaply than a memory.
#
# Cost: 4.1k parameters per layer, 10.3057M against 10.3016M. Initialised to the IDENTITY
# (w[:,0,-1] = 1), so at init it is exactly a no-op -- verified bit-for-bit against conv=0
# in float64 -- and it can only earn its keep from there. Decode == prefill with a random
# filter to 1.8e-15, and the ring buffer carries across split prefills to 8.9e-16.
#
# Then the two arms the ablation asked for, kept behind it:
#   --persist 0.875  reallocate the spectrum toward the half that carries the loss (the
#                    damped half costs 0.13 nats when muted, the persistent half 2.04)
#   --Ls 128         widen the pillar; the window is 64 by convention, not by measurement
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

# kv cut short at 205M: -0.036 median vs round 1, real but not enough.


LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_conv4      --Ls 64  --conv 4
cell d1024_persist875 --Ls 64  --persist 0.875
cell d1024_L128       --Ls 128
echo "##### LONG5H_D1024_L DONE"
