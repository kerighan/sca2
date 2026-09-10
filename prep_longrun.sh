#!/usr/bin/env bash
# Build the corpus the overnight run needs. CPU + network only, no GPU: safe to
# run while anything else is training.
#
# WHY A NEW CORPUS IS NEEDED. Nothing on disk can extend the token horizon past
# what the campaign already used. Measured, not guessed:
#
#   pycode_long1024.pt    177.4M   <- the campaign corpus, and the largest
#   fineweb_300M.pt       298.0M   raw, but a FLAT stream: median doc 667 tokens,
#                                  so a 1024-window spans ~2 documents
#   fineweb_300M longdoc'd at T=1024 -> 170.4M, i.e. LESS than pycode
#   fineweb_long2048.pt   106.3M
#
# And the flat stream is not a usable shortcut: prep_longdoc.py's own docstring
# says an unrelated prefix FLATTERS a decaying state (GDN drops the previous
# document for free) against a non-decaying one (the C head keeps it). Running
# the long horizon on raw fineweb_300M would hand GDN the very axis under test.
#
# So: stream more codeparrot-clean, REUSING THE EXISTING BPE (--bpe) so the
# tokens are identical in kind to the 177.4M already measured. Same corpus
# family, same tokenizer, same long-document filter -- only longer.
#
# 600M raw at pycode's measured 89.6% long-doc retention (177.4/198.0) gives
# ~537M tokens, a 3.2x extension of the current endpoint.
#
# DISK: 59 GB free at time of writing (97% full). The two .pt files below cost
# ~4.5 GB; the HF streaming cache is the risk, so watch it.
#
#   bash prep_longrun.sh 2>&1 | tee runs/prep_longrun.log
set -eu

RAW=pycode_bpe16k_600M
OUT=pycode_long1024_big

echo "### streaming + tokenizing 600M tokens of codeparrot-clean"
python -u prep_pycode.py \
    --tokens 600000000 \
    --min-tokens 1025 \
    --bpe pycode_bpe16k \
    --out "$RAW"

echo "### carving intra-document 1024-token windows"
python -u prep_longdoc.py --src "$RAW.pt" --out "$OUT" --T 1024

echo "### done; sizes:"
python - <<'EOF'
import torch
for f in ("pycode_long1024.pt", "pycode_long1024_big.pt"):
    try:
        z = torch.load(f)
        print(f"  {f:28} train {len(z['train'])/1e6:7.1f}M  val {len(z['val'])/1e6:5.1f}M")
    except FileNotFoundError:
        print(f"  {f:28} MISSING")
EOF
echo "### if the big corpus is under ~450M, lower --tokens expectations in longrun.sh"
