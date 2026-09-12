#!/usr/bin/env bash
# d=1024, arms 3 and 4. Continues after d1024_lapa_M512, whose orchestrator was replaced
# mid-flight (the arm itself kept running, reparented; bash re-reads a running script by
# byte offset, so this is a new file rather than an edit -- SPARK.md §7).
#
# ARM 3, d1024_lapa_ffmatch: round 1's mixer exactly (M=256, dv=256), ff 4096 -> 5740.
# That is a SINGLE change against the d1024_lapa curve already in this log, and it lands
# LapA at 13.670M params/layer against GDN's 13.670M -- +0.00%, 142.9M model either way.
# It removes the one confound round 1 could not answer: the 19% parameter deficit.
#
#   Worth saying in advance so the result is read honestly either way. A standard
#   parameter scaling law puts 19% fewer parameters at about alpha*ln(1.23) ~ 0.02 nats
#   with alpha ~ 0.076, and we are data-limited here (486M tokens for 116M params), which
#   shrinks it further. So this arm is expected to close ~0.02 of the 0.17, not the lot.
#   If it closes much more, the scaling-law reasoning was wrong and the FFN matters more
#   than assumed at this width; if it closes ~nothing, the deficit is ruled out and what
#   remains is structural. Both outcomes are informative, which is why it is worth 5 h.
#
# ARM 4, d1024_lapa_oldamp: the pre-11-September damping (--lam-max 0.125 --damp-mem
# 64,4096) at round 1's shape. Also a single change against d1024_lapa. The window-
# aligned damping has only ever been measured on COPY -- its LM arm was killed twice by
# OOM -- and round 1 ran with it as the default without a contrast arm, so it has never
# been read in an LM. This is that reading.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_lapa_M512" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_lapa_M512 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lapa_ffmatch --ff 5740
cell d1024_lapa_oldamp  --ff 4096 --lam-max 0.125 --damp-mem 64,4096
echo "##### LONG5H_D1024_D DONE"
