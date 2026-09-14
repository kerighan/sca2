#!/usr/bin/env bash
# d=1024: FREE THE MODES. --lam-free with a wide init range, on top of LayerScale.
#
# WHY, measured analytically on the trained 8-layer checkpoint (no training involved):
#
#  * At theta = 0 the long head's read kernel is exactly
#        kappa(Delta) = (1/M) sum_m w_m exp(-(lam_m + i omega_m) Delta)
#    and the head emits (re || im), so a layer contributes exactly TWO real temporal
#    profiles. omega is a FIXED BUFFER, identical in every layer; only lam and w are learned.
#  * The 8 layers' 16 profiles span an EFFECTIVE RANK OF 2.82 (97.7% of the singular mass
#    on 2). The whole stack has the temporal expressivity of one layer.
#  * Why: 86-100% of the |w| mass in every layer sits on one of TWO points -- lam = 0 (the
#    hard `persist` pin, 39-67% of the mass) or lam = lam_max (the clamp, 22-56%). The
#    |w|-weighted median memory is 64.0 tokens = 1/lam_max in 7 layers out of 8.
#  * softplus(a).clamp(max=lam_max) has EXACTLY ZERO gradient above the cap, and the free
#    modes sitting in that dead zone are 103/128, 88/128, 126/128, 126/128, 112/128,
#    128/128, 106/128, 53/128 -- layer 5 has every single free mode trapped, forever.
#  * GDN, measured the same way on a real batch: 64 (layer, head) pairs spanning 2.5 to
#    5.8e6 tokens, 200x to 1.3e6x spread INSIDE each layer, and a monotone depth gradient
#    in the median (12.9, 120, 118, 118, 372, 458, 1054, 7983 -- x619 over the stack).
#  * The LINK FUNCTION is not the problem, and this is worth stating because it was the
#    first hypothesis: for lam <~ 1/64, softplus(a) ~ exp(a), so |d log tau / da| = 0.94 ..
#    1.00 across the whole range. Already scale-free, already the log-rate parameterisation
#    GDN (exp(A_log)), Mamba (-exp(A_log)) and RWKV (exp(-exp(w))) use. The BOUND is the
#    problem. relu^2 would be strictly worse: zero gradient at a = 0 and linear in the RATE.
#  * Retro-prediction that checks out: d1024_lapa_lam16 already tested "move the cap"
#    (1/64 -> 1/16) and was NULL (-0.015 at 249M, inside the +-0.03 floor). Moving the cap
#    moves one of the two points; it does not create a continuum, and it makes EVERY layer
#    short, the opposite of GDN's depth gradient.
#
# WHAT THIS ARM DOES: lam = exp(a), nothing pinned, no cap but the fp32 safety ceiling
# (55/chunk = 0.43, memory floor 2.3 tokens, just past GDN's fastest head at 2.5), and the
# init spread log-uniformly over 4 .. 20000 tokens instead of 64 .. 2048.
#
# ON TOP OF --layer-scale, deliberately, not as a single change. d1024_ls is running and is
# at -0.136 / -0.151 / -0.062 vs base over its first three evals; if it holds, the arm that
# can beat GDN is the one with both. d1024_ls itself measures LayerScale alone, so the
# decomposition is still available. Read this against d1024_ls for the lam_free share and
# against d1024_fast_kv_conv4 for the total.
#
# GATES (repo harness, sca2.iso --self, lam_free on, damp_mem (4,20000)):
#   float64 prefill/decode/split/grad iso  rel 2.9e-16 .. 6.3e-16  (tol 1e-10)
#   float32                                rel 2.0e-07 .. 3.4e-07  (tol 3e-04)
#   lam_free=False bit-for-bit identical to the current layer (0.00e+00)
#   at init: 0/M modes at lambda=0, 0/M at the ceiling, M/M distinct memories,
#            0/M with zero gradient -- against 17/32 dead on the current path.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_ls" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_ls finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lsfree --layer-scale --lam-free --damp-mem 4,20000
echo "##### LONG5H_D1024_Z DONE"
