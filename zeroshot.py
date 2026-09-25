"""Zero-shot accuracy from log-probabilities, for arms of this campaign.

    python zeroshot.py --ckpt runs/ck_zyda.z_dv256.pt runs/ck_zyda.z_gdn.pt

Why this exists. The campaign is decided on validation loss, and loss has a
measured blind spot: dv256 loses 0.094 nats to the GDN baseline while
generating text that is less degenerate and lexically richer on five
concordant indicators. A loop is highly probable under the model, so it is
cheap in perplexity and useless in practice. A downstream task scores what the
model can DO, and arbitrates between the two readings.

Task choice is not a formality at this scale. These arms are 120-142M
parameters trained on 5-7B tokens, and most of the usual suite is at chance
there -- Pythia-160M scores 25% on MMLU, which is exactly chance, and
Winogrande sits within a point of 50%. Measuring those would report noise with
the authority of a number. The three below are the ones that discriminate at
this size:

  LAMBADA   last-word prediction, no options: the most sensitive, and the
            closest to what the loss already measures, so it is the control.
  PIQA      binary physical commonsense, ~60% at this scale, clearly above
            its 50% chance level.
  ARC-Easy  4-way grade-school science, ~35-40% against 25% chance.

Scoring is the standard: sum the log-probability of each candidate
continuation under the model and take the argmax. Accuracy is reported both
unnormalised and normalised by continuation length, because the two disagree
whenever the candidates differ in length and the disagreement is informative
rather than a detail to hide.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from decode import load


def _score(m, tk, ctx: str, cont: str, device: str) -> tuple[float, int]:
    """(sum log p(cont | ctx), number of continuation tokens)."""
    a = tk.encode(ctx).ids
    b = tk.encode(cont).ids
    if not b:
        return -1e9, 1
    ids = torch.tensor([a + b], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(ids)
    lp = F.log_softmax(lg[0].float(), -1)
    # token t is predicted from position t-1
    tot = sum(lp[len(a) + i - 1, b[i]].item() for i in range(len(b)))
    return tot, len(b)


def lambada(m, tk, device, limit):
    from datasets import load_dataset
    d = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    if limit:
        d = d.select(range(min(limit, len(d))))
    hit = 0
    for r in d:
        words = r["text"].rsplit(" ", 1)
        ctx, last = words[0], " " + words[1]
        a = tk.encode(ctx).ids
        b = tk.encode(last).ids
        ids = torch.tensor([a + b], dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(ids)
        pred = lg[0, len(a) - 1:len(a) + len(b) - 1].argmax(-1).tolist()
        hit += int(pred == b)                      # every token of the last word
    return {"acc": hit / len(d)}, len(d)


def _multi(m, tk, device, items):
    """items: (context, [candidates], gold_index). Returns acc and acc_norm."""
    hit = hitn = 0
    for ctx, cands, gold in items:
        sc = [_score(m, tk, ctx, c, device) for c in cands]
        hit += int(max(range(len(sc)), key=lambda i: sc[i][0]) == gold)
        hitn += int(max(range(len(sc)), key=lambda i: sc[i][0] / sc[i][1]) == gold)
    n = len(items)
    return {"acc": hit / n, "acc_norm": hitn / n}, n


def piqa(m, tk, device, limit):
    from datasets import load_dataset
    d = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    if limit:
        d = d.select(range(min(limit, len(d))))
    items = [(r["goal"], [" " + r["sol1"], " " + r["sol2"]], r["label"]) for r in d]
    return _multi(m, tk, device, items)


def arc_easy(m, tk, device, limit):
    from datasets import load_dataset
    d = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    if limit:
        d = d.select(range(min(limit, len(d))))
    items = []
    for r in d:
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        items.append((f"Question: {r['question']}\nAnswer:",
                      [" " + t for t in r["choices"]["text"]],
                      labels.index(r["answerKey"])))
    return _multi(m, tk, device, items)


TASKS = {"lambada": lambada, "piqa": piqa, "arc_easy": arc_easy}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--bpe", default="zyda_bpe32k")
    p.add_argument("--tasks", default="lambada,piqa,arc_easy")
    p.add_argument("--limit", type=int, default=500,
                   help="examples per task; 0 = all. The standard error of an "
                        "accuracy at n=500 is about 2 points, which is the "
                        "resolution to expect")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from tokenizers import ByteLevelBPETokenizer
    tk = ByteLevelBPETokenizer(f"{a.bpe}-vocab.json", f"{a.bpe}-merges.txt")
    names = [t for t in a.tasks.split(",") if t in TASKS]

    out = {}
    for path in a.ckpt:
        m, cfg, cfg_d, V = load(path, a.device)
        label = cfg_d.get("label", path)
        out[label] = {}
        for t in names:
            res, n = TASKS[t](m, tk, a.device, a.limit)
            se = math.sqrt(0.25 / n)               # worst case, p = 0.5
            out[label][t] = (res, n, se)
            print(f"  {label:9s} {t:9s} n={n:4d}  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in res.items())
                  + f"   (se <= {se:.3f})", flush=True)
        del m
        torch.cuda.empty_cache()

    if len(out) == 2:
        (a1, r1), (a2, r2) = out.items()
        print(f"\n{a1} minus {a2}:")
        for t in names:
            d = {k: r1[t][0][k] - r2[t][0][k] for k in r1[t][0]}
            se = math.sqrt(r1[t][2] ** 2 + r2[t][2] ** 2)
            print("  " + f"{t:9s} "
                  + "  ".join(f"{k} {v:+.3f}" for k, v in d.items())
                  + f"   (se of the difference <= {se:.3f})")


if __name__ == "__main__":
    main()
