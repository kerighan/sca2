"""
Build a wikitext-103 corpus with the SAME 16k BPE as pycode, so the vocab
and embedding are identical and results are directly comparable.

wikitext-103 is ~100M tokens of English Wikipedia, entirely different from
codeparrot: natural language, knowledge-heavy, less structural repetition.
If the architecture gap to GDN is task-dependent (too much structure in code,
not enough novelty), wikitext will show it.

    python prep_wikitext.py                       # uses existing pycode BPE
    python prep_wikitext.py --bpe fineweb_bpe16k  # different BPE

The output is wikitext_long1024.pt with the same format as pycode_long1024_xl.pt:
  {"train": tensor, "val": tensor, "vocab": int}
Documents shorter than T+1 are dropped so every T-window is intra-document.
"""
import argparse
import os
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=1024)
    p.add_argument("--bpe", default="pycode_bpe16k",
                   help="BPE prefix (reuse the code BPE for vocab compatibility)")
    p.add_argument("--out", default=None)
    p.add_argument("--val-frac", type=float, default=0.002, dest="val_frac")
    a = p.parse_args()
    out = a.out or f"wikitext_long{a.T}.pt"
    if os.path.exists(out):
        print(f"exists: {out}"); return

    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer

    # Load the same BPE used for pycode
    tk = ByteLevelBPETokenizer(f"{a.bpe}-vocab.json", f"{a.bpe}-merges.txt")
    V = tk.get_vocab_size()
    print(f"BPE: {a.bpe}, vocab {V}")

    # Load wikitext-103
    print("loading wikitext-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Tokenize: concatenate paragraphs within each article, separate articles with EOT
    eot = tk.encode("<|" + "endoftext" + "|>").ids
    all_ids = []
    for split in ("train", "validation", "test"):
        doc = []
        for row in ds[split]:
            text = row["text"]
            if text.strip() == "":
                # Empty line = article boundary in wikitext
                if doc:
                    ids = tk.encode("".join(doc)).ids
                    if ids:
                        all_ids.extend(ids)
                        all_ids.extend(eot)
                    doc = []
            else:
                doc.append(text)
        if doc:
            ids = tk.encode("".join(doc)).ids
            if ids:
                all_ids.extend(ids)
                all_ids.extend(eot)
    print(f"total tokens: {len(all_ids):,}")

    data = torch.tensor(all_ids, dtype=torch.int32)

    # Carve into long documents (reuse prep_longdoc logic)
    from prep_longdoc import carve
    T = a.T
    carved, ndoc = carve(data, T, eot=eot[0] if eot else 0)
    print(f"after carving T={T}: {len(carved):,} tokens from {ndoc} documents")

    # Train/val split
    n_val = max(int(len(carved) * a.val_frac), T * 10)
    n_val = (n_val // T) * T  # align
    val = carved[:n_val]
    train = carved[n_val:]
    print(f"train {len(train):,} / val {len(val):,} tokens")

    torch.save({"train": train, "val": val, "vocab": V}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
