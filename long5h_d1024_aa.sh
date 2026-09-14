#!/usr/bin/env bash
# d=1024: LayerScale + free modes + TWO READ GROUPS on the long head.
#
# WHY NG=2 AND NOT MORE RANK IN THE ABSTRACT. Three read-rank experiments were null
# before, all phrased as "more rank". This one has a specific, measured mechanism:
#
#   The mode bank holds two populations that do different jobs. The fast modes
#   (periods 2..2048) discriminate positions inside the window -- that is ADDRESSING.
#   The 64 slow modes (slow_frac 0.25, periods 2048..20480, i.e. 2x to 20x the context)
#   do not move in phase within the window, so they address NOTHING -- they INTEGRATE,
#   and they are where the LM keeps its document memory.
#
#   With one w, both jobs share one temporal kernel. Measured on the trained checkpoint,
#   what the model chooses is a compromise that costs it dearly:
#
#     couche          0     1     2     3     4     5     6     7
#     |w| sur lents 19.4% 28.7% 29.8% 35.1% 22.6% 24.6% 18.9% 17.1%
#     moy |rho|     .182  .252  .281  .324  .205  .209  .145  .138
#     sans les lents.080  .136  .134  .142  .086  .100  .094  .091
#
#   rho(D) = (1/M) sum_m w_m e^{i om_m D} is the weight with which the read at t picks
#   up the write at t-D, so every non-zero rho(D != 0) is a FALSE MATCH. Dropping the
#   slow modes HALVES the mean sidelobe in all 8 layers. The model pays that to keep
#   document memory, because a single w cannot do both.
#
#   NG=2 splits the value channels in two, each read through its own w, hence its own
#   kernel: kappa_g(D) = (1/M) sum_m w[m,g] e^{-i om_m D}. One group can zero the slow
#   modes (clean addressing), the other can concentrate on them (integration).
#
# POST-HOC SIGNATURE, checkable on the checkpoint without any loss comparison: |w| on
# the slow modes must DIVERGE between the two columns. If both columns look the same,
# the freedom was there and the gradient did not take it -- the same outcome lam_free
# just had, and it would mean the compromise is not what is limiting us.
#
# WHY NOT NG=4: the hypothesis is binary (address vs integrate), and 4 tests "more rank"
# again. Also the grouped path builds Fq as (B,K,NG,2C,2M), so the CODE CONSTRUCTION
# grows with NG while both GEMMs stay NG-invariant (group width compensates) -- and this
# machine is bandwidth-starved (292 FLOP/byte). NG=4 pays speed for a vaguer question.
# If NG=2 pays, NG=4 is the obvious follow-up.
#
# GATES (CPU, nothing on the GPU while d1024_lsfree is up):
#   sca2.iso --self, lam_free + long_groups 2 (+ kv_dk 8, conv 4):
#     float64 prefill/decode/split/grad iso  3.5e-16 .. 8.2e-16  (tol 1e-10)
#     float32                                1.3e-07 .. 3.1e-07  (tol 3e-04)
#   with the two w columns RANDOMISED and distinct, lam_raw randomised over 1/20000..0.4:
#     decode == prefill  6.1e-16 .. 7.2e-16      chunked == single chunk  2.8e-16 .. 3.1e-16
#   NG=2 with the two columns EQUAL is bit-for-bit NG=1 (0.00e+00): identity at init.
#   dvi = dv + kv_dk = 272 = 2 x 136, so the split is exact.
#
# --save-every 3600: five mid-run checkpoints, same path as the final save (which
# overwrites them). d1024_ls was cut 1h30 before its save and its analysis was lost.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_lsfree" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_lsfree finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lsfree_g2 --layer-scale --lam-free --damp-mem 4,20000 --long-groups 2
echo "##### LONG5H_D1024_AA DONE"
