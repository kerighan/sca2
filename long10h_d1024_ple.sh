#!/usr/bin/env bash
# 10h run: gdngate + Ls=128 + PLE (ple_dim=128).
# Queued after l10_gdn in the 10h comparison.
#
# PLE (Per-Layer Embedding) gives each layer its own token-identity signal
# directly, bypassing the residual stream. A shared (V, ple_dim*L) embedding
# is sliced per layer, projected to d, and added to the residual before the
# layer. Cost: 17.8M params (V*128*8 + 128*d*8), bringing the total from
# 120.7M to 138.6M — still 11.5% below GDN's 156.6M.
#
# Why this might help: our deep layers contribute very little (layer 7:
# +0.010 nat when muted). PLE gives them a direct signal from the token
# identity, not filtered through 7 layers of processing. If the problem is
# that deep layers have nothing new to say because the residual stream has
# already absorbed the token, PLE fixes that.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_l10_gdn\.l10_gdn\.pt" runs/long10h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) l10_gdn finished"

LOG=runs/long10h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 36000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell l10_gdngate_ple --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate --ple-dim 128
echo "##### LONG10H_PLE DONE"
