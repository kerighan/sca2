"""
Pretraining-regime benchmark: equal WALL CLOCK, single pass, 2 layers.

Different from bench_tinypython.py in the three ways that matter:

  * SINGLE PASS. Batches are taken sequentially, not sampled with replacement,
    so every token is seen once and the epoch~1 regime is real rather than
    approximate. 40 epochs on 1.6M tokens measures memorization as much as
    learning; this does not.
  * EQUAL WALL CLOCK, not equal steps. Compilation and evaluation are excluded
    from the clock; a warmup runs before it starts.
  * A real web corpus (FineWeb-Edu) with a 16k BPE.

Both arms see identical data in identical order.

    python prep_fineweb.py            # once
    python pretrain.py --seconds 900
"""
import argparse, json, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sca2.ref import LayerCfg
from bench_tinypython import SCA2, Transformer, generate


def batches(data, B, T, device):
    """Sequential, non-repeating."""
    step = B * T
    i = 0
    while i + step + 1 <= len(data):
        chunk = data[i:i + step + 1].long()
        x = chunk[:-1].view(B, T).to(device, non_blocking=True)
        y = chunk[1:].view(B, T).to(device, non_blocking=True)
        i += step
        yield x, y


@torch.no_grad()
def evaluate(m, val, B, T, device, nb=25):
    m.eval()
    ls = []
    for k, (x, y) in enumerate(batches(val, B, T, device)):
        if k >= nb:
            break
        ls.append(F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item())
    m.train()
    return sum(ls) / len(ls)


def run(name, m, tr, va, a, device, log):
    m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    npar = sum(p.numel() for p in m.parameters())
    core = m.core_params() if hasattr(m, "core_params") else 0
    print(f"{name}: {npar} params ({core} in layers)", flush=True)
    it = batches(tr, a.batch, a.block, device)

    for _ in range(a.warm):                       # untimed: absorbs compilation
        x, y = next(it)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
        opt.step()
    torch.cuda.synchronize()

    spent, step, seen, nxt = 0.0, 0, 0, 0.0
    while spent < a.seconds:
        try:
            x, y = next(it)
        except StopIteration:
            print(f"{name}: corpus exhausted at step {step}", flush=True); break
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        loss.backward(); opt.step()
        torch.cuda.synchronize()
        spent += time.perf_counter() - t0
        step += 1; seen += a.batch * a.block
        if spent >= nxt or spent >= a.seconds:
            vl = evaluate(m, va, a.batch, a.block, device)
            rec = {"model": name, "step": step, "train_s": round(spent, 1),
                   "tokens": seen, "train": round(loss.item(), 5), "val": round(vl, 5),
                   "tok_s": round(seen / spent)}
            print(f"  {name} t={rec['train_s']:6.0f}s step {step:6d} "
                  f"{seen/1e6:6.1f}M tok  train {rec['train']:.4f}  val {rec['val']:.4f}  "
                  f"{rec['tok_s']} tok/s", flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            nxt = spent + a.eval_every
    return m


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--eval-every", type=float, default=30, dest="eval_every")
    p.add_argument("--warm", type=int, default=10)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--block", type=int, default=256)
    p.add_argument("--d", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--Mc", type=int, default=64); p.add_argument("--Md", type=int, default=8)
    p.add_argument("--G", type=int, default=8); p.add_argument("--ff", type=int, default=256)
    p.add_argument("--trf-ff", type=int, default=367, dest="trf_ff")
    p.add_argument("--variant", default="v3polar_cc"); p.add_argument("--freq", default="rope")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", default="fineweb_bpe16k.pt")
    p.add_argument("--log", default="runs/pretrain.jsonl")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--save", default=None)
    p.add_argument("--label", default=None, help="name this arm in the log")
    p.add_argument("--only", default=None)
    p.add_argument("--no-compile", action="store_true", dest="no_compile")
    a = p.parse_args(argv)
    device = "cuda"

    z = torch.load(a.data)
    tr, va, V = z["train"], z["val"], z["vocab"]
    print(f"corpus {len(tr)/1e6:.1f}M train / {len(va)/1e6:.1f}M val tokens, vocab {V}")
    print(f"budget {a.seconds:.0f}s per arm, {a.layers} layers, B={a.batch} T={a.block}\n")

    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff, freq=a.freq, theta_scale=0.0, max_len=a.block)
    log = open(a.log, "a")
    models = {}
    if a.only != "transformer":
        torch.manual_seed(0)
        nm = a.label or "SCA2"
        models[nm] = run(nm, SCA2(V, cfg, a.variant, device, a.layers),
                         tr, va, a, device, log)
    if a.only != "sca2":
        torch.manual_seed(0)
        trf = Transformer(V, a.d, 4, a.trf_ff, a.block, a.layers).to(device)
        # The SCA2 arm runs through a compiled variant, so the transformer must be
        # compiled too. It was not -- here and in bench_tinypython.py -- which
        # inflated SCA2's wall-clock results and understated the transformer's
        # margin wherever it already won.
        if not a.no_compile:
            blocks = trf.blocks
            trf = torch.compile(trf, dynamic=False)
            trf.core_params = lambda: sum(p.numel() for p in blocks.parameters())
        models["Transformer"] = run("Transformer", trf, tr, va, a, device, log)
    log.close()
    if a.save:
        for name, m in models.items():
            inner = getattr(m, "_orig_mod", m)
            torch.save({"model": inner.state_dict(), "cfg": vars(a), "V": V},
                       f"{a.save}.{name.lower()}.pt")
            print(f"saved {a.save}.{name.lower()}.pt", flush=True)

    if a.samples:
        from tokenizers import ByteLevelBPETokenizer
        tk = ByteLevelBPETokenizer("fineweb_bpe16k-vocab.json", "fineweb_bpe16k-merges.txt")
        g = torch.Generator().manual_seed(7)
        for name, m in models.items():
            print(f"\n{'='*70}\n{name} samples\n{'='*70}")
            for k in range(a.samples):
                i = int(torch.randint(len(va) - 96, (1,), generator=g))
                pr = va[i:i + 64].long()
                out = generate(m, pr, 120, device)
                print(f"\n--- {k+1} ---\n[PROMPT] {tk.decode(pr.tolist())!r}\n"
                      f"[CONT]   {tk.decode(out.tolist())!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
