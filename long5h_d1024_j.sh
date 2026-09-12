#!/usr/bin/env bash
# d=1024: the KEY-VERIFICATION GATE. The best variant at d=128, never tried at scale.
#
# Why this and not more capacity. Three capacity/parameterisation changes have now moved
# nothing at d=1024: M and dv both doubled (median -0.018 nats, inside noise, 1.7x the
# compute), the decay cap released 4x (median +0.002), and the mixer's read weights show
# the long head IS connected (||W_long|| ~ ||W_short|| in mix, ratio 0.90). Whatever costs
# 0.17 nats is not the size of the memory.
#
# The repo already diagnosed what it is, in arch_cdelta.py's docstring:
#
#   "the C head costs ~0.28 nats on words new to the window, uniformly, and neither the
#    query z (gated_read) nor the read's magnitude (cdelta_raw) tells 'found' from 'not
#    found' -- measured: read norms on new and repeated words are the same distribution.
#    So the gate needs evidence the read itself carries."
#
# cshort_damphkv is that gate. It stores a key beside the value, reads it back, and gates
# the output on whether the read-back key matches the query:
#   e_s = [V(z_s) ; Kv(h_s)],  m_t = cos(Re key part, Kv(z_t)),  g_t = sigmoid(a.m_t + b)
#
# At d=128 in runs/long5h.jsonl it was the BEST arm: l5_B_s0 (cshort_damphkv) -0.0598 nats
# vs GDN, against l5_A_s0 (cshort_damph, i.e. v1) -0.0366. 1.6x the margin, for +0.016M
# parameters per layer at d=1024 (10.318M vs 10.302M). SPARK.md §1 defines v1 as the
# WORSE of the two and §8 files this under "can wait"; the d=1024 results say otherwise.
#
# COST: cshort_damphkv runs through the sca2 mirror, not the optimised lapa/layer.py, so
# expect ~1.9x slower -- roughly 250M tokens in 5 h against round 1's 486M. The comparison
# window narrows but stays above where the d=128 gap became readable.
#
# ARM 1 is a single change against round 1: the variant, nothing else.
# ARM 2 stacks the two corrections the round-1 checkpoint asked for on top of it.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --Mc 256 --dv 256 --Ls 64 --rope-base 1000 --slow-frac 0.25"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_kv      --variant cshort_damphkv_cc --theta-scale 0.02
cell d1024_kv_free --variant cshort_damphkv_cc --theta-scale 0.20 --lam-max 0.0625
echo "##### LONG5H_D1024_J DONE"
