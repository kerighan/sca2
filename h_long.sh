#!/bin/bash
# Long context, T=2048: does the extra distance get used, and by whom?
#
# Corpus: fineweb_long2048.pt (prep_longdoc.py), 106M tokens in which every
# T-aligned window lies INSIDE ONE DOCUMENT. On the unfiltered corpus a T=2048
# window spans ~2.1 documents (median document is 616 tokens), which would test
# tolerance to unrelated preceding text instead of range -- and would flatter a
# decaying state (GDN drops the previous document for free) over a non-decaying
# one (the C head keeps it). One pass, so no arm sees a token twice.
#
# Instrument: --pos-buckets 8, loss by position within the window. The aggregate
# loss cannot decide this (sd = 0.039 nats at 25 eval batches, the size of every
# effect chased today); the per-position profile is relative WITHIN a model, so
# its own noise largely cancels. --eval-batches 100 on top, which the 300 val
# windows now afford.
#
# B=2 keeps 4096 tokens per step, as in every run today, so the step count and
# the optimizer trajectory stay comparable to the T=256 results.
#
# SCA2 arms use the loop-free D head: at T=2048 the old one would run
# T/16 = 128 Python iterations per layer (32 at T=256), which is exactly the
# launch-bound regime measured at 79k vs 97.7k tok/s.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
export SCA2_D_CHUNK=16
echo "pgid $$ -- stop with: kill -- -$$"
C="--only sca2 --seconds 1200 --eval-every 120 --batch 2 --block 2048 --layers 2 \
   --data fineweb_long2048.pt --log runs/hlong.jsonl --samples 0 \
   --eval-batches 100 --pos-buckets 8"

echo "=== $(date +%H:%M) 1/3 SCA2long (Md=4 -> ff=364, loop-free D head) ==="
python -u pretrain.py $C --Mc 128 --Md 4 --ff 364 --freq rope \
  --variant v3polarflat_cc --label SCA2long --save runs/ck_hl_sca2

echo "=== $(date +%H:%M) 2/3 GDNlong ==="
python -u pretrain.py $C --Mc 128 --Md 16 --ff 256 \
  --variant gdn_cc --label GDNlong --save runs/ck_hl_gdn

# Mc is the C head's ADDRESSING width, and rope's grid stays unaliased to 2e4
# positions, so if anything needs more of it at T=2048 it is this arm.
echo "=== $(date +%H:%M) 3/3 SCA2longMc (Md=4 -> Mc=256) ==="
python -u pretrain.py $C --Mc 256 --Md 4 --ff 300 --freq rope \
  --variant v3polarflat_cc --label SCA2longMc --save runs/ck_hl_mc

echo "=== $(date +%H:%M) ALL DONE ==="
