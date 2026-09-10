#!/usr/bin/env bash
# Short probe: does a learned local window replace the hand-made bigram?
#
# The C head's write key is h, and h is a hardcoded shift by one token
# (arch_gatedc.GatedLayer.prefill: h[:,1:] = z[:,:-1]) -- a dense bigram, not
# learned. cfg.conv replaces it with a causal depthwise conv of width k on z,
# identity-initialized, which is what GDN carries as ShortConv(k=4) on q/k/v.
#
# Why this and not more theta: the position profile at one epoch shows GDN ahead
# of SCA2 at EVERY bucket, by 0.111 nats in tokens 0-127 where there is almost no
# context to use. That is a local-machinery deficit, not a memory one. theta 0.02
# moved the profile in the predicted direction (slope -0.277 -> -0.321, past
# GDN's -0.304) but cost 0.033 at 0-127 and was net +0.012 on val.
#
# conv is implemented ONLY in arch_gatedc.GatedLayer, so --conv 4 with
# --variant v3polarflat_cc would be silently ignored. Both arms therefore run in
# the gc family, where conv=0 with every other knob off is iso with v3polar
# (see registry note on gc_plain). ff absorbs conv's 512 params/layer:
#     conv0 ff=364 -> 743,528     conv4 ff=362 -> 743,520
#
# 900s per arm, same seed, so this is equal-WALL-CLOCK not equal-tokens; the
# tok/s printed per arm says whether that matters. A quick go/no-go only -- the
# seed sd on this corpus is 0.03-0.07, so nothing here is conclusive on its own.
#
#   setsid nohup bash conv_probe.sh > runs/conv_probe.log 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/conv_probe.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --dv 32 --Mc 378 --Md 4 --variant gc_cc --c-heads 1
        --seconds 900 --eval-batches 60 --eval-every 150 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --seed 0 --log $LOG"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell cv0 --conv 0 --ff 364
cell cv4 --conv 4 --ff 362

echo "##### CONV PROBE DONE"
