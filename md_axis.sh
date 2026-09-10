#!/usr/bin/env bash
# Is SCA2's LM deficit a starved D head? The Mc -> Md trade at equal parameters.
#
# Every arm run on pycode so far, including all four of confirm_pycode.sh, used
# Md=4. That puts 16,384 floats/layer of state in the C head, which never
# forgets, against 512 in the D head, which is the gated recurrence -- while GDN
# carries 10,800 floats/layer entirely in a delta-rule store. So the hypothesis
# is that Mc buys copy capacity and Md buys induction, and the LM loss comes from
# having spent nearly everything on Mc.
#
# Md is 18x more expensive than Mc per unit (9280 params vs 524 at dv=64), so it
# cannot be raised at fixed dv without crushing ff -- which would confound the
# result with FFN capacity. Halving dv instead halves Md's cost and keeps ff=364.
# Measured: dv=32/Md=16/Mc=166 runs at 70.6k tok/s against 65.0k for
# dv=64/Md=16/Mc=32, with more Mc AND the full ff. So dv=32 is the funding lever.
#
# Two arms, because one would be uninterpretable:
#   md4_dv32   dv 32, Md  4, Mc 378   743,528 params   state 24,192 C / 256 D
#   md16_dv32  dv 32, Md 16, Mc 166   743,800 params   state 10,624 C / 1024 D
# md4 vs md16 isolates the Mc->Md trade at fixed dv, fixed ff, fixed params.
# md4_dv32 vs deep4 (dv 64, Mc 128, Md 4, 2.9188) isolates the dv->Mc trade.
#
# 3600s is a cap; both should exhaust the 177.4M-token corpus first (~2500s),
# making this equal-tokens against confirm_pycode.sh by construction.
#
#   setsid nohup bash md_axis.sh > runs/md_axis.log 2>&1 < /dev/null &
#   python dump_md_axis.py
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16

LOG=runs/md_axis.jsonl
COMMON="--data pycode_long1024.pt --block 1024 --batch 8 --d 128 --layers 4
        --dv 32 --ff 364 --variant v3polarflat_cc
        --seconds 3600 --eval-batches 60 --eval-every 120 --pos-buckets 8
        --samples 0 --only sca2 --freq rope --log $LOG"

cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON "$@"; }

cell md4_dv32   --Md 4  --Mc 378
cell md16_dv32  --Md 16 --Mc 166

echo "##### MD AXIS DONE"
