#!/usr/bin/env bash
# 1.75B raw tokens of codeparrot-clean -> ~1.55B train tokens in 1024-token intra-document
# windows: enough for 5 h per arm (1.4B tokens at 78k tok/s) with NO second epoch.
# Same BPE, same filter, same carve as the 537M corpus. ~25-40 min, CPU only.
set -eu
python -u prep_pycode.py --tokens 1750000000 --min-tokens 1025 --bpe pycode_bpe16k --out pycode_bpe16k_1750M
python -u prep_longdoc.py --src pycode_bpe16k_1750M.pt --out pycode_long1024_xl --T 1024
rm -f pycode_bpe16k_1750M.pt        # 7 GB intermediate; the carved file is what trains
python - <<'PY'
import torch; z=torch.load("pycode_long1024_xl.pt"); print(f"### XL READY train {len(z['train'])/1e6:.1f}M val {len(z['val'])/1e6:.1f}M")
PY
