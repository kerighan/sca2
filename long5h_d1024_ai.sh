#!/usr/bin/env bash
# d=1024: gate_mix — gdn-gate AFTER the mix projection, before the residual add.
# This is the placement closest to GDN's actual readout: GDN gates before its
# output projection o, we gate after ours (mix). Wiki ablation: gate_mix was #2
# behind gate_long (3.0757 vs 3.0248), but it is the more principled placement
# and may behave differently on a longer run with more data.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval --save-every 3600
        --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_gatemix --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate --gdn-gate-scope mix
echo "##### LONG5H_D1024_AI DONE"
