#!/usr/bin/env bash
# Arms 2 and 3 of the d=1024 convergence run: GDN then GDN-2.
#
# GDN runs second (i.e. first of the two baselines) because it is the comparison the
# campaign's headline rests on, and GDN-2 is the more suspect implementation of the
# pair -- if only two of the three arms finish, those two should be the ones that
# settle LapA vs GDN. GDN-2 preflighted clean here (finite loss and grads at the run
# shape, fla's Triton kernel agreeing with fla's own naive reference to 1.3e-3), but
# that is a smoke test, not a statement that the architecture is faithfully reproduced.
#
# Why a second file rather than an edit. long5h_d1024.sh was mid-flight when the order
# changed, and bash re-reads a running script by byte offset -- editing it shifts the
# next command (SPARK.md §7). Its orchestrator was killed while its child (the LapA
# arm) kept running, reparented and still writing to runs/long5h_d1024.log; this script
# waits for that arm's own end-of-run marker and then continues the queue.
#
# Everything else is identical to long5h_d1024.sh: same log, same corpus, same seed,
# same COMMON block, same lr re-derived from the probe by the same rule. Arm order does
# not touch the comparison -- the arms are independent, same seed and same data order
# each -- it only changes which result lands first.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

# pretrain.py prints "saved <--save>.<label>.pt" as its last act for an arm.
until grep -q "saved runs/ck_d1024_lapa" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) arm 1 (d1024_lapa) finished"

LR=$(python - <<'PY'
import json, math
last = {}
for line in open("runs/lr_probe_d1024.jsonl"):
    r = json.loads(line)
    if math.isfinite(r["val"]):
        last[r["model"]] = r["val"]
best = min(last, key=last.get) if last else "lr5e-4"
print(best.replace("lr", ""))
PY
)
echo "##### $(date +%H:%M) lr=$LR (same rule as long5h_d1024.sh)"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr $LR --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_gdn  --variant gdn_cc  --gdn-heads 8 --gdn-head-k 128
cell d1024_gdn2 --variant gdn2_cc --gdn-heads 8 --gdn-head-k 128
echo "##### LONG5H_D1024 DONE"
