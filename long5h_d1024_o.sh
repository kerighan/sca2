#!/usr/bin/env bash
# d=1024: the same two mechanisms, on the OPTIMISED layer.
#
# d1024_kv_conv4 (running) gets the quality answer, but through sca2's mirror at 23.9k
# tok/s. lapa/layer.py is 1.95x faster than that structure and now carries both mechanisms
# itself: --kv-dk 16 is a port of sca2's CHeadDeltaKV, verified exact against it in float64
# -- 8.88e-16 on both prefill paths and in decode, identical parameter count (7579 at the
# gate shape, 10.3221M/layer at d=1024), no unmatched keys.
#
# So d1024_fast_kv_conv4 is the SAME FUNCTION as the arm now running, computed faster. Two
# things come out of it that the current arm cannot give:
#   - the wall-clock reading, which is where LapA's case actually lives. At matched tokens
#     the running arm sits +0.085 nats behind GDN against round 1's +0.191; at equal wall
#     clock that projects to +0.045 against +0.088, and more throughput improves it further.
#   - a cross-check of the port under training rather than at a single float64 point: two
#     implementations, same seed, same data order, curves that should lie on top of each
#     other. If they diverge, the port is wrong somewhere the static check did not reach --
#     which is exactly how the conv's bf16 leak got through this afternoon.
#
# Then L128 on top, widening the pillar the ablation identified (short head muted costs
# +6.45 nats against the long head's +2.77).
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_kv_conv4" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_kv_conv4 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_fast_kv_conv4      --Ls 64
cell d1024_fast_kv_conv4_L128 --Ls 128
echo "##### LONG5H_D1024_O DONE"
