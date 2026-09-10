"""Copy capacity: can a layer reproduce a random string it has just read?

    [BOS] s_1 .. s_L [SEP] s_1 .. s_L          loss and accuracy on the second copy only

The string is fresh random each batch, so no prior helps: the only way to emit
token i is to have carried it, exactly, in the state across the separator. This
reads state capacity and addressing directly. Lengths are mixed during training
and reported separately at eval, so each arm yields a CURVE per length over
training -- the quantity of interest is where (and whether) each length saturates.

Fairness: a recurrent layer carries a fixed-size state, attention carries the
prefix (O(T)) and is a ceiling, not a peer. Among recurrent arms the honest axis
is accuracy vs STATE SIZE; params and state are logged next to every record.

    # one arm per call; several calls append to one log
    python -m lapa.benchmarks.copy run --arm lapa --label "lapa M=256" --d 1024 --M 256 --dv 256 \
        --lengths 128,256,512,1024 --steps 6000 --log runs/copy_d1024.jsonl
    python -m lapa.benchmarks.copy run --arm gdn --label "gdn 8x128" --d 1024 --gdn-heads 8 --gdn-head-k 128 ...
    python -m lapa.benchmarks.copy plot runs/copy_d1024.jsonl --out plot/copy_d1024.png

The plot has one panel per length: accuracy vs training step, one curve per arm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F

from ..model import LM
from .baselines import build


def make_batch(B, L, S, Tmax, device, g):
    """x (B,Tmax) tokens, y (B,Tmax) targets (-100 outside the copy)."""
    BOS, SEP, PAD = S, S + 1, S + 2
    s = torch.randint(0, S, (B, L), generator=g)
    x = torch.full((B, Tmax), PAD, dtype=torch.long)
    x[:, 0] = BOS
    x[:, 1 : 1 + L] = s
    x[:, 1 + L] = SEP
    x[:, 2 + L : 2 + 2 * L] = s
    y = torch.full((B, Tmax), -100, dtype=torch.long)
    y[:, 1 + L : 1 + 2 * L] = s
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(m, lengths, B, S, Tmax, device, nb, seed=1234):
    """per length: token accuracy and exact-string rate."""
    g = torch.Generator().manual_seed(seed)
    m.eval()
    acc, exact = {}, {}
    for L in lengths:
        tok = ex = n = 0
        for _ in range(nb):
            x, y = make_batch(B, L, S, Tmax, device, g)
            pred = m(x).argmax(-1)
            mask = y != -100
            hit = (pred == y) & mask
            tok += hit.sum().item()
            n += mask.sum().item()
            ex += (hit.sum(1) == mask.sum(1)).sum().item()
        acc[str(L)] = round(tok / n, 4)
        exact[str(L)] = round(ex / (nb * B), 4)
    m.train()
    return acc, exact


def layer_kwargs(a):
    if a.arm == "lapa":
        kw = dict(
            M=a.M,
            dv=a.dv,
            L=a.L,
            ff=a.ff,
            theta_scale=a.theta_scale,
            persist=a.persist,
            chunk=a.chunk,
            beta_init=a.beta_init,
        )
        return kw
    if a.arm == "gdn":
        return dict(
            heads=a.gdn_heads, head_k=a.gdn_head_k, expand_v=a.gdn_expand_v, ff=a.ff
        )
    if a.arm == "attn":
        return dict(heads=a.attn_heads, ff=a.ff, max_len=4096)
    raise SystemExit(f"unknown arm {a.arm}")


def run(a):
    device = a.device
    lengths = [int(x) for x in a.lengths.split(",")]
    Tmax = a.tmax or 2 * max(lengths) + 2
    S = a.symbols
    V = S + 3
    torch.manual_seed(a.seed)
    kw = layer_kwargs(a)
    m = LM(lambda i: build(a.arm, a.d, **kw), a.layers, V, a.d).to(device)
    n_layer = m.layer_params() // a.layers
    state = m.state_floats()
    label = a.label or f"{a.arm}"
    print(
        f"=== {label}: {a.layers} layers d={a.d}, {n_layer} params/layer, state {state if state is not None else 'O(T) kv'} floats/seq, "
        f"lengths {lengths}, Tmax {Tmax}, B={a.batch}",
        flush=True,
    )
    fwd = torch.compile(m, dynamic=False) if a.compile else m
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(a.seed)
    log = open(a.log, "a")
    ac = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (a.dtype == "bf16" and device.startswith("cuda"))
        else torch.autocast(device_type="cpu", enabled=False)
    )
    t0 = time.perf_counter()
    seen = 0
    for step in range(1, a.steps + 1):
        L = lengths[(step - 1) % len(lengths)]  # cycle lengths, one per batch
        x, y = make_batch(a.batch, L, S, Tmax, device, g)
        with ac:
            loss = F.cross_entropy(
                fwd(x).flatten(0, 1).float(), y.flatten(), ignore_index=-100
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        seen += a.batch * Tmax
        if step % a.eval_every == 0 or step == a.steps:
            acc, exact = evaluate(
                m, lengths, a.eval_batch, S, Tmax, device, a.eval_batches
            )
            el = time.perf_counter() - t0
            rec = {
                "label": label,
                "arm": a.arm,
                "step": step,
                "s": round(el, 1),
                "tok_s": round(seen / el),
                "train": round(loss.item(), 4),
                "acc": acc,
                "exact": exact,
                "params_layer": n_layer,
                "state": state,
                "d": a.d,
                "layers": a.layers,
                "kw": kw,
                "seed": a.seed,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(
                f"  step {step:6d} {el:6.0f}s train {loss.item():.3f}  "
                + "  ".join(
                    f"L{L}:{acc[str(L)]:.2f}/{exact[str(L)]:.2f}" for L in lengths
                ),
                flush=True,
            )
    log.close()


def plot(a):
    """Writes <out> (token accuracy) and <out stem>_exact.<ext> (exact-string rate)."""
    import os

    stem, ext = os.path.splitext(a.out)
    for metric, out in (("acc", a.out), ("exact", f"{stem}_exact{ext}")):
        _plot_one(a.log, metric, out)


def _plot_one(log, metric, out):
    import collections
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.loads(l) for l in open(log)]
    lengths = sorted({int(k) for r in recs for k in r["acc"]})
    arms = collections.OrderedDict()
    for r in recs:
        arms.setdefault(r["label"], []).append(r)
    n = len(lengths)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False, sharey=True
    )
    cmap = plt.get_cmap("tab10")
    for i, L in enumerate(lengths):
        ax = axes[i // cols][i % cols]
        for j, (lab, rs) in enumerate(arms.items()):
            xs = [r["step"] for r in rs if str(L) in r[metric]]
            ys = [r[metric][str(L)] for r in rs if str(L) in r[metric]]
            st = rs[0]["state"]
            nl = rs[0]["layers"]
            tag = (
                f"{lab}  ({rs[0]['params_layer'] / 1e6:.2f}M params, {st / nl / 1e3:.1f}k state / layer)"
                if st
                else f"{lab}  ({rs[0]['params_layer'] / 1e6:.2f}M params / layer, KV cache)"
            )
            ax.plot(xs, ys, color=cmap(j % 10), lw=2, label=tag if i == 0 else None)
        ax.set_title(f"copy length L = {L}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.set_xlabel("training step")
        if i % cols == 0:
            ax.set_ylabel("exact-string rate" if metric == "exact" else "token accuracy")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    d = recs[0]["d"]
    fig.suptitle(
        f"copy capacity, d={d}, {recs[0]['layers']} layers -- {metric} on the second copy",
        y=1.0,
    )
    axes[0][0].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print("saved", out)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", default="lapa", choices=["lapa", "gdn", "attn"])
    r.add_argument("--label", default=None)
    r.add_argument("--d", type=int, default=128)
    r.add_argument("--layers", type=int, default=2)
    r.add_argument("--ff", type=int, default=448)
    r.add_argument("--M", type=int, default=190)
    r.add_argument("--dv", type=int, default=56)
    r.add_argument("--L", type=int, default=16)
    r.add_argument("--theta-scale", type=float, default=0.02, dest="theta_scale")
    r.add_argument("--persist", type=float, default=0.5)
    r.add_argument("--chunk", type=int, default=128)
    r.add_argument("--beta-init", type=float, default=-2.0, dest="beta_init", help="erase-gate bias; -1000 switches the delta rule off")
    r.add_argument("--gdn-heads", type=int, default=3, dest="gdn_heads")
    r.add_argument("--gdn-head-k", type=int, default=60, dest="gdn_head_k")
    r.add_argument("--gdn-expand-v", type=float, default=1.0, dest="gdn_expand_v")
    r.add_argument("--attn-heads", type=int, default=4, dest="attn_heads")
    r.add_argument("--lengths", default="16,32,64,128")
    r.add_argument("--tmax", type=int, default=0)
    r.add_argument("--symbols", type=int, default=64)
    r.add_argument("--steps", type=int, default=4000)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    r.add_argument("--eval-batches", type=int, default=4, dest="eval_batches")
    r.add_argument("--eval-batch", type=int, default=64, dest="eval_batch")
    r.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    r.add_argument("--compile", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--log", default="runs/copy_lapa.jsonl")
    pl = sub.add_parser("plot")
    pl.add_argument("log")
    pl.add_argument("--out", default="plot/copy.png", help="token-accuracy figure; the exact-string figure is written next to it as *_exact")
    pl.add_argument("--exact", action="store_true", help="no-op, kept so older scripts still run (both figures are always written)")
    a = p.parse_args(argv)
    (run if a.cmd == "run" else plot)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
