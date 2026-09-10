#!/usr/bin/env bash
# Two arms on the SHORT-HEAD base, after catch_short_s0 has finished (waits for the
# COMBO QUEUE DONE marker, not for an idle GPU, so it cannot slip in between cells):
#   A  catch_shortdamp_s0    damped long head (half the modes pinned at lambda=0) + short dft head      ff=448
#   B  catch_shortdampkv_s0  same + key verification                                                    ff=440
# Read: A - short = the damping; B - A = does the gate still earn its keep once the
# noise is treated at the source. Same ff as short (448) and combo (440) so each
# pair differs by the damping alone (+190 params/layer).
#   setsid -f nohup bash long_damp_queue.sh > runs/long_damp_queue.log 2>&1 < /dev/null
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_D_CHUNK=16
until grep -q "COMBO QUEUE DONE" runs/long_combo.log 2>/dev/null; do sleep 60; done
COMMON="--data pycode_long1024_big.pt --block 1024 --batch 8 --d 128 --layers 4
        --Md 4 --G 8 --freq rope
        --seconds 9000 --eval-batches 60 --eval-every 300 --pos-buckets 8
        --samples 0 --only sca2 --log runs/catchup.jsonl --class-eval"
cell () { local label=$1; shift; echo "##### $label :: $*"
          python -u pretrain.py --label "$label" $COMMON --save runs/ck_$label "$@"; }
cell catch_shortdamp_s0   --seed 0 --variant cshort_damph_cc   --Mc 190 --dv 56 --ff 448 --Ls 16 --theta-scale 0.02
cell catch_shortdampkv_s0 --seed 0 --variant cshort_damphkv_cc --Mc 190 --dv 56 --ff 440 --Ls 16 --theta-scale 0.02
echo "##### DAMP QUEUE DONE"
