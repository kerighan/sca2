#!/usr/bin/env bash
# d=1200: PARAMETER-MATCHED against GDN, with NG=2 + triton_scan.
#
# At d=1024, our layer budget is 82.6M vs GDN's 109.4M — a 24.5% deficit.
# That deficit has never been controlled: every arm of this campaign compared
# architectures at DIFFERENT sizes. d1024_lapa_ffmatch tried matching via
# the FFN (ff 4096 -> 5740) and was null, but ff is the wrong axis — it
# adds decode capacity, not mixer capacity, and our mixer is what differs.
#
# d=1200 with ff=4800 (= 4*d) matches GDN's layers at +0.9%:
#   layers: 110,289,064 vs 109,357,184 (+931,880)
#   total:  149,629,448 vs 156,599,696 (-4.5%, embed is smaller)
#
# M=256 and dv=256 are KEPT — the spectral resolution and value width are
# unchanged. What grows: the projections K(d->M), V(d->dv), mix(4dv->d),
# and the FFN (d->ff->d). The mixer's core (the chunked spectral accumulator)
# is UNCHANGED — same state, same codes, same solve. The extra width gives
# the projections a larger receptive field in the residual stream.
#
# triton_scan because NG=2 is free with the fused kernel (263k tok/s on the
# layer, vs 229k batched). Iso: float64 4.9e-16, float32 4.2e-07.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

until grep -q "saved runs/ck_d1024_g4tri" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_g4tri finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1200 --layers 8
        --ff 4800 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_big_g2 --layer-scale --lam-free --damp-mem 4,20000 --long-groups 2
echo "##### LONG5H_D1024_AD DONE"
