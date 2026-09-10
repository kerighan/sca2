#!/bin/bash
# Four arms, 30 min each, matched budget (185,900-186,220 params/layer, GDN -0.7%),
# 2 layers, B=16 T=256, single pass over FineWeb-Edu 298M (no arm exhausts it).
#
# Arms 3 and 4 test the one speed lever that survived a clean measurement
# (sweep_dhead.py, idle GPU): the D head's cost is linear in Md because `vc` in
# its einsum does not depend on m, and Md=16 -> 4 measures x1.36 end to end.
# The 27,840 params/layer that frees are given back to the parts that are almost
# free in time -- the FFN (arm 3) or the C head's addressing (arm 4), the C head
# being ~5 ms of a ~42 ms step. So this is a pure reallocation at fixed budget,
# and the question is whether Md was worth its 66% of the training step.
#
# Equal WALL CLOCK, so the faster arms see more tokens; dump_h4.py reports both
# equal-time and equal-token readings.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
echo "pgid $$ -- stop everything with: kill -- -$$"
C="--only sca2 --seconds 1800 --eval-every 60 --layers 2 \
   --data fineweb_300M.pt --log runs/h4.jsonl --samples 0"

echo "=== $(date +%H:%M) 1/4 SCA2 (Mc=128 Md=16 ff=256) ==="
SCA2_D_CHUNK=8 python -u pretrain.py $C --Mc 128 --Md 16 --ff 256 \
  --variant v3polar_cc --freq rope --label SCA2 --save runs/ck_h4_sca2

echo "=== $(date +%H:%M) 2/4 GDN ==="
SCA2_D_CHUNK=8 python -u pretrain.py $C --Mc 128 --Md 16 --ff 256 \
  --variant gdn_cc --label GDN --save runs/ck_h4_gdn

echo "=== $(date +%H:%M) 3/4 SCA2fast (Md=4 chunk16, params -> ff=364) ==="
SCA2_D_CHUNK=16 python -u pretrain.py $C --Mc 128 --Md 4 --ff 364 \
  --variant v3polar_cc --freq rope --label SCA2fast --save runs/ck_h4_fast

echo "=== $(date +%H:%M) 4/4 SCA2fastMc (Md=4 chunk16, params -> Mc=256) ==="
SCA2_D_CHUNK=16 python -u pretrain.py $C --Mc 256 --Md 4 --ff 300 \
  --variant v3polar_cc --freq rope --label SCA2fastMc --save runs/ck_h4_fastmc

echo "=== $(date +%H:%M) ALL DONE ==="
