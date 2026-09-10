#!/usr/bin/env bash
# Confirm the complex delta rule at n=3, and separate its TWO mechanisms.
#
# The screen (cdelta.sh, n=1, full epoch) came back large and, unlike wg2,
# self-consistent -- the aggregate val and the position profile agree, and the
# gate actually moved (beta bias 0.11->0.27, |w| 0 -> 0.8..1.75, so erasure
# became data-dependent rather than a constant):
#
#            val      vs md4_dv32 (sd 0.036)     slope
#   cdelta_t0    2.7653   -0.1405  (3.9x)        -0.420
#   cdelta_t02   2.7447   -0.1610  (4.5x)        -0.464
#   gdn4  n=3    2.7869                          -0.302
#   md4_dv32 n=3 2.9058                          -0.260
#
# Two things now need resolving, and they are different questions:
#
# 1. IS IT REAL. n=1 against a control sd of 0.036 only screens. Two more seeds
#    per arm gives n=3, the minimum that can call it.
#
# 2. WHICH MECHANISM. t0 and t02 differ by 0.020, well inside the control sd, so
#    on this evidence the CONTENT-dependent part of the erasure does nothing and
#    the whole gain would be the theta=0 effect: G is then the Gram matrix of
#    the rope dictionary, which is documented near-degenerate (rank 49/128 at
#    T=128, ref.freq_grid), and the triangular solve is its Gram-Schmidt
#    whitening. That is a claim about REDUNDANCY OF THE CODE BOOK, not about
#    erasing associations -- a different paper. n=3 on both arms is what
#    separates "delta rule" from "whitening a degenerate basis".
#
# Note what would NOT follow from a t0 == t02 tie: that erasure is useless. At
# theta=0 the erase still fires, it is just addressed by LAG instead of by
# content. Distinguishing those needs the dft arm (where theta=0 makes the solve
# an exact no-op, G=I) and that is the natural follow-up, not this run.
#
# Beating GDN is NOT claimed here: cdelta_t02 - gdn4 = -0.042 against a GDN seed
# sd of 0.073, i.e. 0.6 sd. The strong result is against md4_dv32, whose sd is
# half as large. n=3 will say whether parity with GDN holds.
#
#   setsid nohup bash cdelta_seeds.sh > runs/cdelta_seeds.log 2>&1 < /dev/null &
#   python dump_cdelta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

while pgrep -f "pretrain.py --label cdelta" > /dev/null; do sleep 30; done

LOG=runs/cdelta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --seconds 4200 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG
        --dv 32 --Mc 378 --Md 4 --ff 364 --variant cdelta_cc"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

# Seed 0 of both arms is already in runs/cdelta.jsonl. Interleaved so that a
# partial run still yields one extra seed on EACH arm rather than three on one.
cell cdelta_t02_s1 --seed 1 --theta-scale 0.02 --save runs/ck_cdelta_t02_s1
cell cdelta_t0_s1  --seed 1 --theta-scale 0.0  --save runs/ck_cdelta_t0_s1
cell cdelta_t02_s2 --seed 2 --theta-scale 0.02 --save runs/ck_cdelta_t02_s2
cell cdelta_t0_s2  --seed 2 --theta-scale 0.0  --save runs/ck_cdelta_t0_s2

echo "##### CDELTA SEEDS DONE"
