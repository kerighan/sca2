#!/usr/bin/env bash
# CATCHUP.md: on WHICH tokens does GDN catch up? gen3 and GDN under the longrun.sh
# protocol (big corpus, 537M tokens, the regime where the crossing happens at
# ~300M) with --class-eval, so every eval carries the loss by token class
# (sca2/tokclass.py). Seed 0 reproduces long_gen3_s0 / long_gdn_s0 to eval noise,
# which is also the check that --class-eval changed nothing. ~2 h per arm.
#
# The bp arms that were here are gone: cdelta_bp lost 0.15-0.38 nats to gen3 at
# n=1 (10x the seed sd) by destroying long-range retrieval -- see CATCHUP.md.
#
#   setsid nohup bash catchup.sh > runs/catchup.log 2>&1 < /dev/null &
#   python dump_catchup.py --log runs/catchup.jsonl --ref catch_gdn_s0
#   python diag_tokclass.py runs/ck_catch_gen3_s0.*.pt runs/ck_catch_gdn_s0.*.pt
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
while pgrep -f "pretrain.py --label" > /dev/null; do sleep 30; done
LOG=runs/catchup.jsonl
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval"
SHAPE="--Mc 190 --dv 56 --ff 364 --theta-scale 0.02"
cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON --save runs/ck_$label "$@"; }
cell catch_gen3_s0 --seed 0 --variant cdelta_cc $SHAPE
cell catch_gdn_s0  --seed 0 --variant gdn_cc --ff 260
echo "##### CATCHUP DONE"
