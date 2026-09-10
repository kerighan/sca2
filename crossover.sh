#!/bin/bash
# Throughput against context length: where does O(T) overtake O(T^2)?
#
# Every speed number quoted today was at T=256 (runs/tps.log: transformer 127.9k,
# gdn_cc 98.8k, v3polar_cc 97.7k). That is the length that flatters attention:
# its quadratic term is still small next to the linear projections, so the
# Transformer was winning on the CONSTANT, not on the complexity. The whole point
# of a linear-attention layer is invisible at T=256.
#
# Tokens per batch are held at 4096, as in every run today, so B falls as T rises
# (T=4096 means B=1, which under-uses the GPU -- read that column with care).
#
# Arms: the current best (Md=4 + loop-free D head), the pre-today SCA2 to check
# the speedup survives at long T, GDN, and the Transformer.
#
# Caveat: bench_train_tps.py pre-stages its tensors on the GPU, and that was
# measured to overstate SCA2 by 19% against the real loop (97.7k vs 79.0k) --
# though the loop-free D head is precisely what closed that gap. Relative shape
# across T is the point here, not the absolute level.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
export SCA2_D_CHUNK=16
echo "pgid $$ -- stop with: kill -- -$$"

for TB in "256 16" "512 8" "1024 4" "2048 2" "4096 1"; do
  set -- $TB
  echo "=== T=$1  B=$2  ($(( $1 * $2 )) tokens/batch) ==="
  python -u bench_train_tps.py --block $1 --batch $2 --Mc 128 --Md 4 --ff 364 \
    --arms v3polarflat_cc,v3polar_cc,gdn_cc,transformer 2>&1 \
    | grep -viE "warn|not enough sms"
done
echo "=== DONE ==="
