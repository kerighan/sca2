#!/usr/bin/env bash
# d=1024, arms 3-4. Replaces the ff-matched arm with the clean isolation of M.
#
# ARM 3, d1024_lapa_M1024: M=1024, dv=256 -- dv back to round 1, all the extra budget in
# ADDRESSING. Three things make it the right arm rather than more of the same:
#   - it isolates M, which is what the round-1 diagnostic pointed at (the loss was worst
#     on words repeated inside the window) and what the running M512 arm confounded by
#     moving dv at the same time;
#   - its state, 550k, is within 3% of the running arm's 566k, so the pair asks a sharp
#     question: at EQUAL state, does addressing or value width matter?
#   - M is the cheap axis in parameters -- 11.09M/layer against the M512 arm's 12.14M,
#     because dv drives mix(4dv->d), which is over half the mixer.
#
# ARM 4, d1024_lapa_oldamp: unchanged, the pre-11-September damping against the aligned
# default at round 1's shape. The alignment has only ever been read on copy.
#
# Dropped: d1024_lapa_ffmatch (ff 4096 -> 5740, parameter-matched to GDN). It closed the
# 19% parameter confound cleanly, but a standard scaling law only puts that confound at
# ~0.02 of the 0.17 nats, and it would have cost throughput (bigger FFN) in exchange --
# improving the matched-token reading while worsening the equal-wall-clock one. Worth
# running eventually; not worth 5 h ahead of the isolation of M.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_lapa_M512" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_lapa_M512 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lapa_M1024 --Mc 1024
cell d1024_lapa_oldamp --Mc 256 --lam-max 0.125 --damp-mem 64,4096
echo "##### LONG5H_D1024_E DONE"
