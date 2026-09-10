#!/bin/bash
# Launch-overhead experiment, measured where the problem actually shows up.
#
# The microbenchmark and the training loop disagree for SCA2 (97.7k vs 79.0k
# tok/s) and agree for GDN (98.8k vs 96.9k), so the thing to measure is short
# pretrain.py runs, not bench_train_tps.py: only pretrain.py has the CPU busy
# loading data while the D head issues its ~1000 launches per forward.
#
# 2x2, all four the SAME FUNCTION (iso-verified against v3polar):
#   cc / cg     compile, without / with cudagraphs on the prefill
#   v3polar     32-iteration Python chunk loop
#   v3polarflat loop removed, two-level closed form (sca2/fast_dhead.py)
# then the best of those combined with Md=4, the arithmetic lever.
#
# 120 s per config, val loss ignored -- this measures tok/s only.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
echo "pgid $$ -- stop with: kill -- -$$"
C="--only sca2 --seconds 120 --eval-every 60 --layers 2 --data fineweb_300M.pt \
   --log runs/speed2.jsonl --samples 0 --freq rope"

for v in v3polar_cc v3polarflat_cc v3polar_cg v3polarflat_cg; do
  echo "=== $v  Md=16 chunk8 ==="
  SCA2_D_CHUNK=8 python -u pretrain.py $C --Mc 128 --Md 16 --ff 256 \
    --variant $v --label "$v.Md16" 2>&1 | grep -E "tok/s|Error|error"
done

for v in v3polarflat_cc v3polarflat_cg; do
  for ch in 16 32 64; do
    echo "=== $v  Md=4 chunk$ch ==="
    SCA2_D_CHUNK=$ch python -u pretrain.py $C --Mc 128 --Md 4 --ff 364 \
      --variant $v --label "$v.Md4.c$ch" 2>&1 | grep -E "tok/s|Error|error"
  done
done

echo "=== reference: GDN ==="
python -u pretrain.py $C --Mc 128 --Md 16 --ff 256 --variant gdn_cc --label GDN.ref \
  2>&1 | grep -E "tok/s|Error|error"
echo "=== DONE ==="
