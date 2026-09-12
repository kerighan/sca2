#!/usr/bin/env bash
# d=1024: consolidate the win, then add ONE thing.
#
# d1024_max bundled three changes on top of kv+conv and came in +0.076 nats WORSE than its
# own base, median over 40-143M, positive at every point. Cut at 143M. The cost of
# bundling is that we cannot say which of --kv-gate-pc, --beta-groups 3 or --ff 5729 did
# it. One suspect stands out: --beta-groups 3 hands the SLOW INTEGRATORS their own erase
# gate, and those modes are persistent (lambda = 0) and carry 55-85% of the trained LM's
# state energy per SPARK.md -- the document memory. A shared beta protected them behind an
# average; a dedicated one lets the gradient push it up and wipe exactly what they
# accumulate. Unverified, but it is the only one of the three with a mechanism of failure.
#
# ARM 1, d1024_fast_kv_conv4: the exact port of the best arm into lapa/layer.py, ff back to
# 4096, nothing else. Three things at once, all of them consolidation rather than gambling:
#   - it validates the kv port under TRAINING, not at one float64 point. The curve should
#     lie on d1024_kv_conv4's. If it does not, the port is wrong somewhere the static check
#     did not reach -- which is exactly how the conv's bf16 leak got through.
#   - it recovers ~12% throughput (lapa/layer.py is 1.95x the sca2 mirror's structure),
#     which lands directly in the equal-wall-clock reading -- our best number, +0.049.
#   - it becomes a clean, fast base to stack one mechanism at a time on.
#
# ARM 2 adds --kv-gate-pc alone: the only one of d1024_max's three changes for which I have
# no mechanism of failure, and the one the reference architectures both support (GDN and
# sca2's own gated_read are per channel; ours applied one scalar to 2*dv).
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

cell d1024_fast_kv_conv4    
cell d1024_fast_kv_conv4_pc --kv-gate-pc
echo "##### LONG5H_D1024_R DONE"
