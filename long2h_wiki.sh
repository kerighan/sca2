#!/usr/bin/env bash
# Wikitext-103 comparison: gdngate vs GDN vs lapa baseline.
# 48M train tokens = ~3 epochs in 5h at 26k tok/s. 2h = ~1.2 epoch, enough to see.
# Same BPE (pycode_bpe16k, V=16384), same T=1024, same batch=8.
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/wiki.jsonl
: > "$LOG"  # fresh log

LAPA="--variant lapa_cc --Mc 256 --dv 256 --Ls 64 --theta-scale 0.02
      --rope-base 1000 --slow-frac 0.25 --conv 4
      --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate"

COMMON="--data wikitext_long1024.pt --block 1024 --batch 8 --d 1024 --layers 8
        --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
        --seconds 7200 --eval-batches 30 --eval-every 300 --pos-buckets 8
        --samples 0 --log $LOG --class-eval --save-every 3600"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $COMMON --save runs/ck_$label "$@"; }

# ARM 1: our best config (gdngate)
cell wiki_gdngate --only sca2 $LAPA

# ARM 2: GDN with Triton kernels
cell wiki_gdn --only sca2 --variant gdn_cc --gdn-heads 8 --gdn-head-k 128 --gdn-expand-v 1.0

# ARM 3: lapa baseline (no gdngate, with kv_dk, our round-1 equivalent)
cell wiki_lapa_base --only sca2 --variant lapa_cc --Mc 256 --dv 256 --Ls 64 \
     --theta-scale 0.02 --rope-base 1000 --slow-frac 0.25 --conv 4 --kv-dk 16 \
     --layer-scale --lam-free --damp-mem 4,20000

# ARM 4: Mamba2 (SSD) via fla, expand=1 (11.8M/layer, between us and GDN)
cell wiki_mamba2 --only sca2 --variant mamba2_cc --mamba-expand 1

echo "##### WIKI DONE"
