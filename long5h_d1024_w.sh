#!/usr/bin/env bash
# d=1024: DATA-DEPENDENT FORGETTING -- make the layers stop being interchangeable.
#
# The per-layer ablation on the 10-layer checkpoint is what motivates this. Muting each
# layer's mixer in turn costs:
#   layer 0 +1.590 | 1 +0.438 | 2-6 +0.11..0.20 | 7 +0.029 | 8 +0.013 | 9 +0.056
# The deep layers are nearly inert. That is why --layers 10 came in at -0.003 against the
# 8-layer base over 33 points, and why every capacity lever has been null: the model does
# not use more of what it already has, because each layer does much the same thing as the
# last and the marginal one adds nothing.
#
# lam is a CONSTANT per mode, frozen after training, identical in character from layer to
# layer. GDN's is g_t = -exp(A_log)*softplus(a(x_t)+dt_bias), a function of the token, and
# ablates at +1.08 there. Making forgetting content-dependent is the one remaining
# structural difference that could let layers specialise -- each could learn a different
# regime of when to forget rather than a different set of fixed rates.
#
# It stays inside the Laplace frame: the poles become time-varying. The chunked closed form
# survives because with C_t = sum_{u<=t} lam_u the decay is exp(-(C_t - C_s)), so the ramps
# become a cumsum along the chunk and the factorisation gw_s = exp(C_s), gq_t = exp(-C_t)
# is unchanged -- the same thing GDN does with chunk_local_cumsum.
#
# Verified rather than assumed: at init (Wd = 0) it reproduces the learn_persist form to
# 8.88e-16; with a RANDOMISED Wd decode == prefill at 9.99e-16 on both prefill paths and
# split prefills match at 8.88e-16, which is what checks the cumsum factorisation; lam
# spans the full (0, lam_max) range; lam_max*chunk = 2.0 against the float32 limit near 60;
# decay_input=False leaves every standing gate at 8.88e-16.
#
# Costs d*M = 262k parameters per layer (10.584M vs 10.322M).
#
# Cut to make room: d1024_beta0 (--beta-init 0.0, GDN's erase gate starts at 0.5 and ours
# at 0.12) was running a median +0.011 / mean -0.017 against the base at 185M -- no
# direction, the signature of a null.
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

cell d1024_decayin --decay-input
echo "##### LONG5H_D1024_W DONE"
