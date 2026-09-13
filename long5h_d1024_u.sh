#!/usr/bin/env bash
# d=1024: SiLU after the conv -- the layer's only non-linearity outside the FFN.
#
# Our causal conv was a PURELY LINEAR convolution. GDN's q/k/v convs are
# F.silu(F.conv1d(...)), and the user reports this mattered a lot on the original SCA.
# Until now the whole path from z to the residual was linear: conv -> phase codes -> GEMMs
# -> _rms -> mix. The only non-linearity in the mixer was _rms itself.
#
# It also reframes three failures rather than contradicting them. Every attempt to lift the
# rank bound by adding LINEAR profiles lost or did nothing -- wg2 (+0.065, long head,
# d=128), gc_ablate's 4 heads (+0.044), sg4 (-0.007, short head, d=1024). If what is
# missing is a non-linearity rather than more linear terms, all three should fail exactly
# as they did, and this should not.
#
# NOTE it breaks identity-at-init, unlike every other mechanism added this campaign: the
# conv starts as the identity, so the layer now starts from silu(z) instead of z. It can
# therefore hurt from the first step, which none of the others could.
#
# dv is back to 256: d1024_dv512 was flat. Its slope came in at -0.4489 +- 0.0133 against
# the base's -0.4527 +- 0.0455, a 0.1 sigma difference -- the visual "catching up" was the
# initial offset closing, which every curve does early. Cut at 289M. That closes the value
# pipe as a hypothesis: 12.6M parameters and 19% of throughput for nothing.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_convsilu --conv-silu
cell d1024_L10      --layers 10
echo "##### LONG5H_D1024_U DONE"
