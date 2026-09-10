"""
Build a LONG-DOCUMENT Python corpus: codeparrot-clean + a 16k BPE.

Why a new corpus instead of just raising --block on TinyPython. TinyPython's
median document is 85 tokens and its longest is 461, so at T=512 a window
already spans ~6 unrelated functions and at T=1024 the long-document filter
would keep ZERO documents. Training there does not measure long-range retrieval
at all; worse, it measures tolerance to an unrelated prefix, which FLATTERS a
decaying state (GDN drops the previous function for free) over a non-decaying
one (SCA2's C head keeps it). That is the opposite of the property being
compared, and it is the likely reason GDN won the T=512 TinyPython run.

Real Python files are the fix: codeparrot-clean documents are whole modules of
several thousand tokens, where a name bound at the top is used far below.

Why a 16k BPE and not cl100k: same reason as prep_fineweb.py -- at d=128 a 100k
vocab puts 25.7M parameters in the embedding against ~0.37M in the layers, so
the loss would measure the softmax instead of the sequence mixer. The BPE is
fitted on code here, not reused from FineWeb: web-text merges would waste the
vocabulary on prose and split indentation badly.

Documents shorter than --min-tokens are dropped here rather than by
prep_longdoc.py, so the token budget is not spent on files that the long-window
carve would throw away anyway.

    python prep_pycode.py --tokens 120000000 --min-tokens 1025
    python prep_longdoc.py --src pycode_bpe16k.pt --out pycode_long1024 --T 1024
"""
import argparse
import os
import sys
import time

import torch

CACHE = "pycode_bpe16k"
DATASET = "codeparrot/codeparrot-clean"
FIELD = "content"
# Assembled from pieces so the literal never appears in this file; some tooling
# treats it as a real end-of-text marker and truncates whatever follows.
EOT_TOKEN = "<|" + "endoftext" + "|>"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=120_000_000)
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--train-chars", type=int, default=40_000_000, dest="train_chars",
                   help="text used to fit the BPE")
    p.add_argument("--min-tokens", type=int, default=1025, dest="min_tokens",
                   help="drop documents shorter than this (T+1 for the carve)")
    p.add_argument("--val-tokens", type=int, default=2_000_000, dest="val_tokens")
    p.add_argument("--out", default=CACHE)
    p.add_argument("--bpe", default=None, help="reuse an existing BPE prefix")
    a = p.parse_args(argv)
    if os.path.exists(a.out + ".pt"):
        print("cache exists:", a.out + ".pt")
        return 0

    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer

    ds = load_dataset(DATASET, split="train", streaming=True)
    it = iter(ds)
    t0 = time.perf_counter()

    # ---- pass 1: the tokenizer, fitted on code (not on web text) ----------- #
    buf, n = [], 0
    if a.bpe and os.path.exists(a.bpe + "-vocab.json"):
        tok = ByteLevelBPETokenizer(a.bpe + "-vocab.json", a.bpe + "-merges.txt")
        print(f"reusing BPE {a.bpe}, vocab {tok.get_vocab_size()}", flush=True)
    else:
        while n < a.train_chars:
            buf.append(next(it)[FIELD])
            n += len(buf[-1])
        print(f"BPE training text: {n/1e6:.0f} MB in {len(buf)} docs "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(buf, vocab_size=a.vocab, min_frequency=2,
                                special_tokens=[EOT_TOKEN])
        tok.save_model(".", a.out)
        print(f"BPE fitted, vocab {tok.get_vocab_size()} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- pass 2: encode in batches, reusing the fitting text first --------- #
    eot = tok.token_to_id(EOT_TOKEN)
    assert eot == 0, f"expected the eot id to be 0 for prep_longdoc, got {eot}"
    ids, total, kept, seen = [], 0, 0, 0
    last_report = 0

    def push(texts):
        """Encode a batch, keeping only documents long enough to carve."""
        nonlocal total, kept, seen
        for e in tok.encode_batch(texts):
            seen += 1
            v = e.ids
            if len(v) + 1 < a.min_tokens:
                continue
            kept += 1
            ids.append(torch.tensor(v + [eot], dtype=torch.int32))
            total += len(v) + 1

    B = 512
    for i in range(0, len(buf), B):
        push(buf[i:i + B])
        if total >= a.tokens:
            break
    buf.clear()
    while total < a.tokens:
        batch = []
        for _ in range(B):
            try:
                batch.append(next(it)[FIELD])
            except StopIteration:
                break
        if not batch:
            print("stream exhausted", flush=True)
            break
        push(batch)
        if total - last_report >= 10_000_000:
            last_report = total
            print(f"  {total/1e6:.0f}M tokens, kept {kept}/{seen} docs "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

    data = torch.cat(ids)[:a.tokens]
    nv = min(a.val_tokens, len(data) // 10)
    torch.save({"train": data[:-nv], "val": data[-nv:],
                "vocab": tok.get_vocab_size()}, a.out + ".pt")
    print(f"saved {a.out}.pt  train {len(data)-nv} val {nv} "
          f"vocab {tok.get_vocab_size()}  kept {kept}/{seen} docs "
          f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
