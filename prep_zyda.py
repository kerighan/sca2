"""
Build a Zyda-2 pretraining corpus: a 32k BPE fitted on the data, tokens
streamed to a uint16 .bin, and a .json sidecar describing it.

Why Zyda-2 and not another slice of code. Every result so far is on
codeparrot, where the token distribution is narrow and highly structural:
every architecture we tried lands on the same wall-clock curve, which is the
compute-optimal frontier being flat rather than the architectures being
equivalent. Zyda-2 is a real pretraining mixture (DCLM + FineWeb-Edu +
Zyda-1 + Dolma-CC, deduplicated and filtered), so a result there transfers.
`sample-100BT` is the canonical 100B-token sample -- far more than we need,
streamable, and stable across runs.

Why a fitted 32k BPE and not tiktoken. tiktoken has no 32k encoding (50k,
100k and 200k only), and at d=1024 with 8 layers a 100k vocab puts 54% of the
parameters in the embedding, so the loss would measure the softmax rather
than the sequence mixer. 32k keeps the embedding at 27% (measured) and is the
size Llama and Mistral use. Fitting it on Zyda-2 itself rather than reusing
the code BPE matters: web-text merges and Python-indentation merges are not
interchangeable.

Why uint16 and not a torch .pt. At 20B tokens a .pt of int32 tensors is 80 GB
and has to be materialised in RAM before saving. uint16 is exact for any
vocab under 65536, halves the file, and streams: this writes as it encodes and
never holds more than one batch. pretrain.py memory-maps the result.

Why PACKED by default, not carved. prep_longdoc.py carves T-token windows that
stay inside one document, which is right for codeparrot (whole modules, several
thousand tokens) and wrong here: web documents have a median around 600 tokens,
so carving at T=2048 would discard most of the corpus. Packing -- concatenate
with an EOT between documents, slice at T -- is what real pretraining does and
what published numbers are comparable to. Use --carve to get the other
behaviour; be aware it flatters a decaying state, which forgets the previous
document for free.

    python prep_zyda.py --tokens 20_000_000_000 --out zyda32k
    python prep_zyda.py --tokens 2_000_000_000 --out zyda32k_2B --bpe zyda_bpe32k

Outputs <out>.bin (uint16 token stream) and <out>.json (meta), plus
<bpe>-vocab.json / <bpe>-merges.txt the first time.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

DATASET = "Zyphra/Zyda-2"
CONFIG = "sample-100BT"
FIELD = "text"
# Assembled from pieces so the literal never appears in this file; some tooling
# treats it as a real end-of-text marker and truncates whatever follows.
EOT_TOKEN = "<|" + "endoftext" + "|>"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=20_000_000_000)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--train-chars", type=int, default=500_000_000, dest="train_chars",
                   help="text used to fit the BPE (a 32k vocab needs more than a 16k one)")
    p.add_argument("--val-tokens", type=int, default=10_000_000, dest="val_tokens")
    p.add_argument("--out", default="zyda32k")
    p.add_argument("--bpe", default="zyda_bpe32k",
                   help="BPE prefix; reused if <prefix>-vocab.json exists, else fitted and saved")
    p.add_argument("--carve", type=int, default=0,
                   help="keep only documents of >= T+1 tokens and emit whole T-token windows "
                        "inside one document (0 = pack, the default)")
    p.add_argument("--batch", type=int, default=1024, help="documents per encode_batch")
    a = p.parse_args(argv)

    bin_path, meta_path = a.out + ".bin", a.out + ".json"
    if os.path.exists(meta_path):
        print("exists:", meta_path)
        return 0

    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer

    t0 = time.perf_counter()
    ds = load_dataset(DATASET, name=CONFIG, split="train", streaming=True)
    it = iter(ds)

    # ---- pass 1: the tokenizer, fitted on this corpus --------------------- #
    buf = []
    if os.path.exists(a.bpe + "-vocab.json"):
        tok = ByteLevelBPETokenizer(a.bpe + "-vocab.json", a.bpe + "-merges.txt")
        print(f"reusing BPE {a.bpe}, vocab {tok.get_vocab_size()}", flush=True)
    else:
        n = 0
        while n < a.train_chars:
            buf.append(next(it)[FIELD])
            n += len(buf[-1])
        print(f"BPE training text: {n/1e6:.0f} MB in {len(buf)} docs "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(buf, vocab_size=a.vocab, min_frequency=2,
                                special_tokens=[EOT_TOKEN])
        tok.save_model(".", a.bpe)
        print(f"BPE fitted, vocab {tok.get_vocab_size()} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    V = tok.get_vocab_size()
    assert V < 65536, f"vocab {V} does not fit in uint16"
    eot = tok.token_to_id(EOT_TOKEN)
    assert eot == 0, f"expected the eot id to be 0, got {eot}"

    # ---- pass 2: encode and stream to disk -------------------------------- #
    # Written as it is produced: at 20B tokens nothing here may accumulate.
    total = kept = seen = 0
    last_report = 0
    T = a.carve
    out = open(bin_path, "wb")

    def push(texts):
        """Encode a batch and append its tokens to the file."""
        nonlocal total, kept, seen
        chunks = []
        for e in tok.encode_batch(texts):
            seen += 1
            v = e.ids
            if T:
                # whole T-token windows inside this document, nothing across
                nwin = len(v) // T
                if nwin < 1:
                    continue
                kept += 1
                chunks.append(np.asarray(v[:nwin * T], dtype=np.uint16))
            else:
                kept += 1
                chunks.append(np.asarray(v, dtype=np.uint16))
                chunks.append(np.asarray([eot], dtype=np.uint16))
        if chunks:
            block = np.concatenate(chunks)
            out.write(block.tobytes())
            total += len(block)

    # the text already pulled for the BPE is used first, then the stream continues
    for i in range(0, len(buf), a.batch):
        push(buf[i:i + a.batch])
        if total >= a.tokens:
            break
    buf.clear()

    while total < a.tokens:
        batch = []
        for _ in range(a.batch):
            try:
                batch.append(next(it)[FIELD])
            except StopIteration:
                break
        if not batch:
            print("stream exhausted", flush=True)
            break
        push(batch)
        if total - last_report >= 100_000_000:
            last_report = total
            el = time.perf_counter() - t0
            rate = total / el
            eta = (a.tokens - total) / rate if rate else 0
            print(f"  {total/1e9:6.2f}B tokens  kept {kept}/{seen} docs  "
                  f"{rate/1e6:.1f}M tok/s  elapsed {el/60:.0f}m  eta {eta/60:.0f}m",
                  flush=True)
    out.close()

    # trim the tail so the file is exactly `tokens` long when we overshot
    if total > a.tokens:
        with open(bin_path, "r+b") as f:
            f.truncate(a.tokens * 2)
        total = a.tokens

    n_val = min(a.val_tokens, total // 20)
    meta = {
        "bin": os.path.basename(bin_path),
        "dtype": "uint16",
        "vocab": V,
        "total": total,
        "n_val": n_val,          # the LAST n_val tokens are validation
        "n_train": total - n_val,
        "dataset": f"{DATASET}:{CONFIG}",
        "bpe": a.bpe,
        "packed": T == 0,
        "carve_T": T,
        "docs_kept": kept,
        "docs_seen": seen,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {bin_path} ({total*2/1e9:.1f} GB) and {meta_path}")
    print(f"  train {meta['n_train']/1e9:.2f}B  val {n_val/1e6:.0f}M  vocab {V}  "
          f"{'packed' if T == 0 else f'carved T={T}'}  "
          f"kept {kept}/{seen} docs  ({(time.perf_counter()-t0)/60:.0f}m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
