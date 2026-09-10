#!/usr/bin/env bash
# Does the C head's CONTENT phase close the gap to GDN?
#
# pw = K(h)*theta + p*omega. At theta=0 the score collapses to a function of the
# lag alone (ref.freq_grid docstring), so the C head -- which holds 24,192 of the
# 24,448 state floats per layer -- is a Fourier-parameterised convolution, not an
# associative memory. Measured, on the variant actually used here:
#
#     theta_scale   |dL/dK|
#     0.0           0.000e+00     <- content path provably dead
#     0.02          1.657e-05
#     0.1           8.104e-05
#
# Every pycode run so far had theta_scale=0.0 hardcoded in pretrain.py, so the
# content path has never once been trained. What it should become when live: the
# write phase uses K(h_s) with h_s = z_{s-1}, the read uses K(z_t), and the value
# stored at s is V(z_s). So the read retrieves V(z_s) for the s where z_{s-1}
# matches z_t -- current token matches the PREDECESSOR of an earlier position,
# return what followed it. That is an induction head, in phase space.
#
# theta is an nn.Parameter(M) in both cases (zeros vs randn*scale), so the
# parameter count is unchanged and the comparison stays matched.
#
# theta=0 is NOT rerun: that is md4_dv32, n=3, already in runs/md_axis.jsonl and
# runs/seeds.jsonl (mean 2.9083, sd 0.0334). Six epochs here, ~3.6h.
#
#   setsid nohup bash theta.sh > runs/theta.log 2>&1 < /dev/null &
#   python dump_theta.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/theta.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --dv 32 --Mc 378 --Md 4 --ff 364 --variant v3polarflat_cc
        --seconds 3600 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

for s in 0 1 2; do
  cell "th002_s$s" --seed "$s" --theta-scale 0.02
done
for s in 0 1 2; do
  cell "th010_s$s" --seed "$s" --theta-scale 0.1
done

echo "##### THETA DONE"
