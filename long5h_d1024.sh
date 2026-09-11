#!/usr/bin/env bash
# THE d=1024 CONVERGENCE RUN: LapA v1 vs Gated DeltaNet vs GDN-2, 5 h each.
# SPARK.md §8 item 2. Equal wall clock, single pass, identical data order per arm.
#
# Shape (SPARK.md §4, and the aspect ratio of LFM2-1.2B which §4 cites as the
# reference point: d/layers = 128):
#   d=1024, 8 layers, block 1024, batch 8, ff 4096 on EVERY arm.
#   LapA M=256 dv=256 L=64  -> 10.30M/layer, mixer 1.90M, state 156k, model 116.0M
#   GDN  8x128              -> 13.67M/layer, mixer 5.27M, state 140k, model 142.9M
#   GDN2 8x128              -> 15.23M/layer, mixer 6.83M, state 140k, model 155.4M
# NOT parameter-matched, deliberately: SPARK.md §3(c)/§4 asks for the three columns
# side by side rather than one matched axis, and no shape matches all three. LapA is
# the SMALLEST model here by 19-25%, which is conservative against it on quality.
#
# Why these settings differ from long5h.sh (d=128), all of them applied to every arm:
#  --amp bf16  : the d=128 campaign trained fp32. At d=1024 fp32 costs ~2.2x on GB10
#                and the layer's deviation under bf16 is ~3e-3 (SPARK.md §9).
#  constant lr : NOT --cosine. Cosine here decays on the fraction of the CORPUS seen,
#                and the arms consume different token counts in the same wall clock
#                (LapA ~459M, GDN ~349M), so a corpus-fraction schedule would hand the
#                arms DIFFERENT learning rates at the same step -- a confound straight
#                through the middle of the comparison. Constant lr has none.
#  --warmup 100: cheap insurance at this width; the campaign used none.
#  --clip 1.0  : the campaign had NO gradient clipping. Observed grad norm at init
#                here is 0.6-0.8 on all three arms, so a clip at 1.0 is inactive in
#                normal operation and only catches a spike -- which is the failure
#                that would cost an unattended 5 h arm. Applied to every arm alike.
#  --Ls 64     : SPARK.md §1 defines v1 with L=64. long5h.sh used 16.
#  GDN kernel  : fla's TRITON path now (SCA2_GDN_KERNEL=naive forces the old
#                reference). Racing the naive reference is what made the d=128 speed
#                numbers wrong by 1.66x -- see SPARK.md §9.
#
# Expected: ~15 h total. Log runs/long5h_d1024.jsonl, read with
#   python plot_lm.py runs/long5h_d1024.jsonl --ref "d1024_gdn"
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "XL READY" runs/prep_xl.log 2>/dev/null; do sleep 60; done
until grep -q "LR_PROBE DONE" runs/lr_probe_d1024.log 2>/dev/null; do sleep 60; done

# lowest final val among the non-diverged probe arms; see lr_probe_d1024.sh
LR=$(python - <<'PY'
import json, math, collections
last = {}
for line in open("runs/lr_probe_d1024.jsonl"):
    r = json.loads(line)
    if math.isfinite(r["val"]):
        last[r["model"]] = r["val"]
best = min(last, key=last.get) if last else "lr5e-4"
print(best.replace("lr", ""))
PY
)
echo "##### $(date +%H:%M) chosen lr=$LR (from runs/lr_probe_d1024.jsonl)"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr $LR --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_lapa --variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02 \
                --rope-base 1000 --slow-frac 0.25
cell d1024_gdn  --variant gdn_cc  --gdn-heads 8 --gdn-head-k 128
cell d1024_gdn2 --variant gdn2_cc --gdn-heads 8 --gdn-head-k 128
echo "##### LONG5H_D1024 DONE"
