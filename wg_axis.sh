#!/usr/bin/env bash
# The WG axis: how many spectral mixtures does the C head read need?
#
# At theta=0 the C head's pre-RMS linear path has dim span{H_l} <= 2 (see
# sca2/arch_wgroup.py): 378 modes and a 24,192-real state give only TWO shared
# temporal profiles, because kappa is a scalar shared by all dv coordinates.
# WG gives each of WG value groups its own w_{m,g}. State and phases unchanged,
# cost 2.Mc.(WG-1) params, absorbed in ff:
#
#     WG=1 ff=364 (== v3polarflat == md4_dv32, the control, n=3 already logged)
#     WG=2 ff=361   WG=4 ff=355   WG=8 ff=343
#
# Scalarity is what allowed ONE score matrix, so the quadratic form now needs one
# per group: measured 49.9k tok/s at WG=4 against 80.0k at WG=1, i.e. 1.6x.
#
# Order is deliberate: bracket the axis at n=1 first (2, 4, 8) and only then
# spend seeds on whichever point wins. A monotone trend across three points is
# worth more than a second seed at one point -- same logic as the Mc/Md/dv axes.
# Nothing here is conclusive alone: seed sd on this corpus is 0.03-0.07.
#
# Waits for the running WG=4 arm to finish, then continues.
#
#   setsid nohup bash wg_axis.sh > runs/wg_axis.log 2>&1 < /dev/null &
#   python dump_wg.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  echo "##### waiting for pid $WAIT_PID (WG=4, seed 0)"
  tail --pid="$WAIT_PID" -f /dev/null
fi

LOG=runs/wgroup.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --dv 32 --Mc 378 --Md 4 --seconds 900 --eval-batches 60 --eval-every 150
        --pos-buckets 8 --samples 0 --only sca2 --freq rope --seed 0 --log $LOG"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell wg2 --variant wg2_cc --ff 361
cell wg8 --variant wg8_cc --ff 343

echo "##### WG AXIS DONE"
