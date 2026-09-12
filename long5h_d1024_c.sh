#!/usr/bin/env bash
# d=1024, round 2. Round 1 had LapA v1 (M=256, dv=256) +0.17 nats behind GDN, worst on
# words repeated inside the window -- in-context retrieval, which is what the long head's
# addressing is for. Two arms, both single changes against curves already in this log.
#
# ARM 1, the attempt: step the mixer up one notch, M=512 dv=512. Still SMALLER than GDN
# on both axes it is judged against -- 12.14M/layer vs 13.67M, mixer 3.74M vs 5.27M --
# so a win reads cleanly. It pays 4x the decode state (566k vs 140k): the honest cost.
#
# ARM 2 fills what the other agent calls the most visible hole in the dossier. The
# window-aligned damping (lam_max = 1/L, mem_range = (L, 32L)) has only ever been
# measured on COPY; its LM arm was killed twice by OOM and does not exist. It has been
# the DEFAULT since 11 September, so round 1 already ran with it -- verified here,
# lam_max = 1/64 and mem_range = (64, 2048) -- but with no contrast arm, so it has never
# actually been read. This runs the pre-11-September setting at round 1's exact shape:
# same M, dv, ff, window, only (lam_max, mem_range) differs, against d1024_lapa.
#
# Not included: --rope-min-period. That knob exists now but is my hypothesis about the
# grid's FAST end, not the damping alignment, and it is unmeasured and not obviously
# right (the rope grid is a positional encoding; dropping high frequencies coarsens
# resolution rather than freeing modes). It has no business inside an arm meant to be
# read as a single change.
#
# Numerical check for both arms: lam_max * chunk = 2.0 at L=64, chunk 128, against the
# float32 closed-form limit of ~88. Safe.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Ls 64 --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lapa_M512 --Mc 512 --dv 512
cell d1024_lapa_oldamp --Mc 256 --dv 256 --lam-max 0.125 --damp-mem 64,4096
echo "##### LONG5H_D1024_C DONE"
