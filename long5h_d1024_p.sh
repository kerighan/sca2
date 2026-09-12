#!/usr/bin/env bash
# d=1024: the optimised layer, both mechanisms, and the gate made per-channel.
#
# The kv gate is the only long-head mechanism that has helped, and its granularity was
# never a measured choice. GDN gates PER CHANNEL -- silu(gp(x)) over H*dv -- and so does
# sca2's own gated_read, over 2*dv. Ours applied ONE number to all 2*dv channels.
# --kv-gate-pc makes ga and gb vectors of length 2*dv: same evidence m, its own slope and
# bias per channel. 1022 parameters at dv=256, and identical to the scalar gate at init
# (verified: broadcasting the reference's scalars into the vectors reproduces sca2's
# cshort_damphkv to 8.88e-16, exactly as the scalar version does).
#
# It is the one thing GDN's readout has that ours does not and that is NOT refuted here.
# The SOURCE of the gate signal is refuted: gen3 + gated_read (input-driven, GDN-style)
# measured +0.090 nats and lost the lead outright, because _rms() erases the very signal
# that says whether anything matched -- a novel key reads as a small random mixture, an
# exact repeat as a full-size value, and RMS maps both to unit scale. The gate has to read
# the RETRIEVAL, which is what kv does. Only the granularity was left on the table.
#
# Arm 1 also carries the port of the gate into lapa/layer.py (exact vs sca2, 8.88e-16 on
# both prefill paths and decode), which is worth ~12% throughput and therefore shows up in
# the wall-clock reading -- the one that matters for this architecture's case.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=batched

until grep -q "saved runs/ck_d1024_kv_conv4" runs/long5h_d1024.log 2>/dev/null; do sleep 60; done
echo "##### $(date +%H:%M) d1024_kv_conv4 finished"

LOG=runs/long5h_d1024.jsonl
COMMON="--data pycode_long1024_xl.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 18000 --eval-batches 60 --eval-every 600 --pos-buckets 8
        --samples 0 --only sca2 --log $LOG --class-eval
        --variant lapa_cc --Mc 256 --dv 256 --theta-scale 0.02
        --rope-base 1000 --slow-frac 0.25 --kv-dk 16 --conv 4"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

cell d1024_fast_kvpc_conv4      --Ls 64  --kv-gate-pc
cell d1024_fast_kvpc_conv4_L128 --Ls 128 --kv-gate-pc
echo "##### LONG5H_D1024_P DONE"
