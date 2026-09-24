#!/usr/bin/env bash
# Gate scope ablation on wikitext — where should the per-channel silu gate go?
# Four arms, 30 min each (1 epoch on 48M tokens). Same base: gdngate + Ls=128.
#
#   long   = long head only (what we've tested so far)
#   both   = long head AND short head (separate gp per head)
#   concat = after concat [ul, us] before mix (one gate over 4*dv=1024 channels)
#   mix    = after mix projection, before residual add (gate over d=1024 channels)
set -u
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SCA2_CTX_CHUNK=128 SCA2_LONG_PATH=triton_scan

LOG=runs/wiki_gate.jsonl
: > "$LOG"

BASE="--data wikitext_long1024.pt --block 1024 --batch 8 --d 1024 --layers 8
      --ff 4096 --Md 4 --G 8 --freq rope --amp bf16 --lr 5e-4 --warmup 100 --clip 1.0
      --seconds 1800 --eval-batches 30 --eval-every 180 --pos-buckets 8
      --samples 0 --only sca2 --log $LOG --class-eval
      --variant lapa_cc --Mc 256 --dv 256 --Ls 128 --theta-scale 0.02
      --rope-base 1000 --slow-frac 0.25 --conv 4
      --layer-scale --lam-free --damp-mem 4,20000 --gdn-gate"

cell () { local label=$1; shift; echo "##### $(date +%H:%M) $label :: $*"
          python -u pretrain.py --label "$label" --seed 0 $BASE "$@"; }

cell gate_long    --gdn-gate-scope long
cell gate_both    --gdn-gate-scope both
cell gate_concat  --gdn-gate-scope concat
cell gate_mix     --gdn-gate-scope mix
echo "##### WIKI_GATE DONE"
