#!/usr/bin/env bash
# d=1024: rank on the pillar, then the value pipe -- both on the best base to date.
#
# BASE = d1024_fast_kv_conv4 (lapa_cc, kv-dk 16, conv 4, M=256 dv=256 Ls=64 ff=4096),
# which is +0.046 nats from GDN at equal wall clock with 24% fewer parameters and a 2.7x
# smaller mixer. Everything below is a single change against it.
#
# ARM 1  --short-groups 4
#   The short head's read weights are (L,): ONE content-dependent 64-tap filter applied
#   identically to all 256 value channels. The taps are the DFT of those weights, so that
#   is a single shared temporal profile -- the rank bound -- and it worsens with width: one
#   filter for dv=56 at d=128, one for dv=256 here. G=4 gives the PILLAR four profiles.
#   The ablation says the short head is what carries this model (+6.45 nats when muted,
#   against the long head's +2.77), and the conv argues the same way from the other side:
#   4 taps PER CHANNEL bought half the gap to GDN, which is not reach (the comb already has
#   64 exact taps) but per-channel diversity, i.e. rank.
#   Not the refuted experiment: wg2 (+0.065) grouped the LONG head, and gc_ablate's 4-head
#   arm (+0.044) ran on v3polar -- three generations back, no delta rule, no short head,
#   80M tokens, d=128. Nobody has grouped the short head.
#   MEASURED COST on an idle GPU: 1.037x, i.e. 426M tokens in 5 h against the base's 442M,
#   and +384 parameters. My earlier +20% estimate was wrong -- the short head is bound by
#   memory traffic, not FLOPs, so 4x the GEMM work on a small matrix costs almost nothing.
#
# ARM 2  --dv 512
#   The value pipe, isolated at last. V projects d -> dv = 1024 -> 256, a 4x CONTRACTION;
#   GDN's v projection is 1024 -> 1024 with no contraction, and Mamba expands to 2d. Every
#   token's contribution to our memory passes a rank-256 bottleneck first. dv=512 halves
#   that. It was tried once, COUPLED WITH M=512, and the pair did nothing (-0.018, inside
#   noise) -- so dv alone has never been read, and it is the axis that was never the
#   suspect. Note this is the expensive axis in parameters (mix is Linear(4*dv, d)):
#   +12.6M on the model, and 1.194x the time, i.e. 370M tokens in 5 h.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_sg4    --dv 256 --short-groups 4
cell d1024_dv512  --dv 512
echo "##### LONG5H_D1024_S DONE"
