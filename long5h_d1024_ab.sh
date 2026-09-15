#!/usr/bin/env bash
# d=1024: the "tuned init" arm. Same architecture as d1024_lsfree_g2, plus:
#
#   --ls-mix-init 0.25   two independent runs converge gs_mix to median ~0.26
#   --ls-ff-init 0.5     both converge gs_ff to median ~0.60
#   --ls-mix-pc          per-channel gs_mix (d,): direction, not just amplitude
#   --w-antipodal 0.10   symmetry-breaking noise on wr/wi: w0=1+eps*n, w1=1-eps*n
#                        mean unchanged, gradient already anti-correlated (cos=-0.71),
#                        but the symmetric point takes 4h to reach cos(kappa)~0.86;
#                        eps=0.10 starts there instantly
#
# The architecture is IDENTICAL to g2 (same --lam-free, --long-groups 2, --layer-scale,
# same --damp-mem, same everything). What changes is ONLY initialisation: where parameters
# start, not what they can express. So reading g2 vs this is a pure init comparison.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_g2init --layer-scale --lam-free --damp-mem 4,20000 --long-groups 2 \
     --ls-mix-init 0.25 --ls-ff-init 0.5 --ls-mix-pc --w-antipodal 0.10
echo "##### LONG5H_D1024_AB DONE"
