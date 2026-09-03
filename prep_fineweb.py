"""
Build a single-pass pretraining corpus: FineWeb-Edu + a 16k BPE.

Why a 16k BPE and not cl100k. At d=128 a 100k vocab would put 25.7M parameters
in the embedding and head against ~318k in the layers -- 99% of the model, and
the loss would measure the softmax rather than the sequence mixer. That is the
exact defect the TinyPython benchmark had. A 16k vocab keeps embedding+head at
4.2M, which still dominates but leaves the layer difference legible, and it
keeps d=128 comparable with everything measured so far. Widening the model is
the other fix, and the better one -- for later, on bigger hardware.

Streams, tokenizes once, caches to disk so every arm sees identical data.
"""
import argparse, os, sys, time
import torch

CACHE = "fineweb_bpe16k"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=80_000_000)
    p.add_argument("--vocab", type=int, default=16384)
    p.add_argument("--train-chars", type=int, default=60_000_000,
                   help="text used to fit the BPE")
    p.add_argument("--out", default=CACHE)
    p.add_argument("--bpe", default=None, help="reuse an existing BPE prefix")
    a = p.parse_args(argv)
    if os.path.exists(a.out + ".pt"):
        print("cache exists:", a.out + ".pt"); return 0

    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    import os.path as _p

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)
    it = iter(ds)

    # ---- pass 1: the tokenizer ----
    t0 = time.perf_counter()
    buf = []
    reuse = a.bpe and _p.exists(a.bpe + "-vocab.json")
    if reuse:
        # reuse an existing BPE so a larger corpus stays token-identical to the
        # smaller one -- otherwise the two runs are not comparable at all
        tok = ByteLevelBPETokenizer(a.bpe + "-vocab.json", a.bpe + "-merges.txt")
        print(f"reusing BPE {a.bpe}, vocab {tok.get_vocab_size()}", flush=True)
    else:
        n = 0
        while n < a.train_chars:
            buf.append(next(it)["text"])
            n += len(buf[-1])
        print(f"BPE training text: {n/1e6:.0f} MB in {len(buf)} docs "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        tok = ByteLevelBPETokenizer()
        tok.train_from_iterator(buf, vocab_size=a.vocab, min_frequency=2,
                                special_tokens=["<|endoftext|>"])
        tok.save_model(".", a.out)
        print(f"BPE fitted, vocab {tok.get_vocab_size()} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- pass 2: encode, reusing the fitting text then continuing the stream ----
    eot = tok.token_to_id("<|endoftext|>")
    ids, total = [], 0
    def push(txt):
        nonlocal total
        e = tok.encode(txt).ids
        ids.append(torch.tensor(e + [eot], dtype=torch.int32))
        total += len(e) + 1
    for d in buf:
        push(d)
        if total >= a.tokens:
            break
    buf.clear()
    while total < a.tokens:
        push(next(it)["text"])
        if total % 5_000_000 < 2000:
            print(f"  {total/1e6:.1f}M tokens ({time.perf_counter()-t0:.0f}s)", flush=True)
    data = torch.cat(ids)[:a.tokens]
    n_val = 2_000_000
    torch.save({"train": data[:-n_val], "val": data[-n_val:], "vocab": tok.get_vocab_size()},
               a.out + ".pt")
    print(f"saved {a.out}.pt  train {len(data)-n_val} val {n_val} vocab {tok.get_vocab_size()} "
          f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
