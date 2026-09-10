#!/usr/bin/env bash
# Tonight: the copy bench's two wins, in the LM, under the 5 h XL protocol (references already
# in runs/long5h.jsonl: l5_A_s0 = A at rope 1e4 / window 16, l5_gdn_s0 = GDN).
#   l5_Arope_s0     A + rope base 1e3                          (185959+258 = same params as A)
#   l5_Aropew64_s0  A + rope base 1e3 + short window 64, ff 424 (186337, +0.06%)
# Waits for the copy combo arm to finish.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
until grep -q "COPY D128 COMBO DONE" runs/copy_d128_combo.log 2>/dev/null; do sleep 60; done
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log runs/long5h.jsonl --class-eval"
cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" $COMMON --save runs/ck_$label "$@"; }
cell l5_Arope_s0    --seed 0 --variant cshort_damph_cc --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02 --rope-base 1000
cell l5_Aropew64_s0 --seed 0 --variant cshort_damph_cc --Mc 190 --dv 56 --ff 424 --Ls 64 --theta-scale 0.02 --rope-base 1000
echo "##### LONG5H ROPE DONE"
