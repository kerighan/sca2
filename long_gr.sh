#!/usr/bin/env bash
# gen3 + GATED READ-OUT, longrun protocol, queued behind the hybrid. CATCHUP.md:
# word_new stays +0.15..+0.25 behind GDN whatever the key geometry (bp), the D head
# (hyb) or the FFN size. On a token whose content is NEW to the window the C head's
# hash read returns an arbitrary mixture of stored values -- and _gated_out RMS-
# normalises it to UNIT scale with no data-dependent gate, so the layer cannot say
# "nothing to retrieve here". GDN's read has an output gate (gp). --gated-read adds
# the same gate to both heads: u * silu(rgate(z)). ff 364 -> 252 pays for it
# (185847 vs 185959, -0.06%). One change vs catch_gen3_s0.
#   setsid nohup bash long_gr.sh > runs/long_gr.log 2>&1 < /dev/null &
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
SHAPE="--Mc 190 --dv 56 --ff 252 --theta-scale 0.02 --gated-read"
echo "##### catch_gen3gr_s0 :: --seed 0 --variant cdelta_cc $SHAPE"
python -u pretrain.py --label catch_gen3gr_s0 --seed 0 --variant cdelta_cc $COMMON $SHAPE --save runs/ck_catch_gen3gr_s0
echo "##### LONG_GR DONE"
