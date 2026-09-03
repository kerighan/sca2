#!/bin/bash
# Two 3-hour single-pass runs at matched parameters (185,984 vs 185,936 per layer),
# 2 layers, on 300M FineWeb-Edu tokens. Both arms compiled. Samples at the end.
cd /media/maixent/2To/sca2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SCA2_CTX_CHUNK=128
COMMON="--seconds 10800 --eval-every 120 --layers 2 --data fineweb_300M.pt \
        --log runs/overnight.jsonl --samples 4"
echo "=== $(date +%H:%M) SCA2 (Mc=128 Md=16) ==="
python -u pretrain.py --only sca2 --Mc 128 --Md 16 --variant v3polar_cc --freq rope \
  $COMMON --save runs/ck_night_sca2
echo "=== $(date +%H:%M) TRANSFORMER (ff=464) ==="
python -u pretrain.py --only transformer --trf-ff 464 \
  $COMMON --save runs/ck_night_trf
echo "=== $(date +%H:%M) ALL DONE ==="
