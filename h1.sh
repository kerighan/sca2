#!/bin/bash
# SCA2 vs Gated DeltaNet, matched budget (~186k params/layer), 2 layers, B=16 T=256.
#
# 1h/arm at the (post-CUDA-fix) ~95k tok/s exhausts the 298M-token corpus in
# ~52 min, so each arm is a FULL SINGLE PASS: equal tokens and equal time, since
# bench_train_tps.py measures the two arms at the same speed (95k vs 98k).
# Third arm: the soft-init C-head decay, the only knob from the gc ablation whose
# sign was not negative (-0.018 at 900s, inside noise) -- this gives it 3.7x the
# tokens to show an effect.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
C="--only sca2 --Mc 128 --Md 16 --seconds 3600 --eval-every 120 --layers 2 \
   --data fineweb_300M.pt --log runs/h1.jsonl --samples 0"
echo "=== $(date +%H:%M) SCA2 (v3polar_cc, rope) ==="
python -u pretrain.py $C --variant v3polar_cc --freq rope --label SCA2 --save runs/ck_h1_sca2
echo "=== $(date +%H:%M) GDN ==="
python -u pretrain.py $C --variant gdn_cc --label GDN --save runs/ck_h1_gdn
echo "=== $(date +%H:%M) SCA2 + soft decay ==="
python -u pretrain.py $C --variant gc_cc --c-decay --c-decay-init soft \
  --label SCA2decay --save runs/ck_h1_decay
echo "=== $(date +%H:%M) ALL DONE ==="
