"""
Carve a long-context corpus out of the existing one, so that a T-token window
is a window INTO ONE DOCUMENT.

Why this is needed. In fineweb_300M.pt the median document is 616 tokens and the
mean is 973, so at T=2048 only 8% of documents even reach the window length and
a window spans ~2.1 documents. Training there would mostly measure tolerance to
whatever precedes an unrelated document -- and it would flatter a decaying state
(GDN forgets the previous document for free) over a non-decaying one (the C head
keeps it), which is the opposite of the property we want to compare.

What this does. Keep only documents of at least T+1 tokens and emit floor((L-1)/T)
windows of exactly T tokens from each. The output is a flat stream whose T-aligned
slices are all intra-document, which is exactly what pretrain.py's `batches()`
produces: it advances by B*T from 0, so every row of every (B,T) batch is one
window.

No re-tokenization: this reads the already-encoded .pt, so the tokens are
identical to the runs done on the unfiltered corpus and the two are comparable.

Known artifact. `batches()` takes T+1 tokens to build (x, y), so the TARGET for
the last position of a row is the first token of the next window -- a different
document. That is one target in T (0.05% at T=2048), identical for every arm, and
the per-position metric in pretrain.py drops the final position anyway.

    python prep_longdoc.py --T 2048
    python prep_longdoc.py --T 2048 --src fineweb_300M.pt --out fineweb_long2048
"""
import argparse
import torch


def carve(d, T, eot=0):
    """Flat stream of T-token windows, each entirely inside one document."""
    pos = (d == eot).nonzero().flatten()
    starts = torch.cat([torch.tensor([0]), pos + 1])          # first token of each doc
    ends = torch.cat([pos, torch.tensor([len(d)])])           # exclusive, eot excluded
    out, ndoc = [], 0
    for s, e in zip(starts.tolist(), ends.tolist()):
        L = e - s
        n = (L - 1) // T                                      # need one token after
        if n < 1:
            continue
        ndoc += 1
        out.append(d[s:s + n * T])
    return (torch.cat(out) if out else d[:0]), ndoc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="fineweb_300M.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--T", type=int, default=2048)
    p.add_argument("--eot", type=int, default=0)
    a = p.parse_args()
    out = a.out or f"fineweb_long{a.T}"

    z = torch.load(a.src)
    res = {"vocab": z["vocab"]}
    for split in ["train", "val"]:
        w, ndoc = carve(z[split], a.T, a.eot)
        res[split] = w
        print(f"{split}: {len(z[split])/1e6:7.1f}M -> {len(w)/1e6:7.1f}M tokens "
              f"({100*len(w)/len(z[split]):4.1f}%), {ndoc} docs, "
              f"{len(w)//a.T} windows of T={a.T}")
    torch.save(res, out + ".pt")
    print(f"saved {out}.pt")


if __name__ == "__main__":
    main()
