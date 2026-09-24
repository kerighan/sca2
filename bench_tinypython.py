"""
SCA2 vs Transformer — token-level TinyPython benchmark.

Dataset: BertilBraun/TinyPython (streamed)
Tokenizer: tiktoken cl100k_base
Task: causal LM over task_description + Python function
Models: 1-layer SCA2 (C+D) vs 1-layer causal Transformer

The SCA2 layer is no longer defined here -- it comes from the `sca2` package,
where every version is gated on numerical equivalence with the frozen reference
(`python -m sca2.iso`). Pick one with SCA2_VERSION=v0|v1|v2 (default: latest);
they are drop-in for each other and share a state_dict.
`sca2/_original_bench.py` keeps a byte-exact snapshot of the layer as first
written, and `python -m sca2.test_fidelity` proves the two agree.

Install:
    pip install torch datasets tiktoken

Run:
    python bench_tinypython.py --steps 3000 --examples 20000 --block 128 --batch 16
    SCA2_VERSION=v0 python bench_tinypython.py        # original speed, same maths
    python bench_tinypython.py --compact-vocab        # see the note it prints

The script caches tokenized data locally after the first run.
"""
import argparse, hashlib, json, math, time, os
import torch
import torch.nn as nn
import torch.nn.functional as F

from sca2 import LayerCfg, make_layer, active_version
from sca2.registry import build

PROMPT = "Write a Python function for the following task:\n\n"
ENCODING = "cl100k_base"


def prepare(args):
    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding(ENCODING)
    eos = enc.eot_token
    # The cache key covers everything that changes the tokens. Keyed on the
    # example count alone, editing PROMPT or the encoding silently reused a
    # stale cache.
    key = hashlib.sha1(f"{ENCODING}|{PROMPT}|{args.examples}".encode()).hexdigest()[:12]
    cache = f"tinypython_{ENCODING}_{args.examples}_{key}.pt"
    # the pre-fix filename, which this PROMPT/encoding is known to have produced
    legacy = f"tinypython_cl100k_{args.examples}.pt"
    src = cache if os.path.exists(cache) else (legacy if os.path.exists(legacy) else None)
    if src:
        if src == legacy:
            print(f"reusing legacy cache {legacy} (same prompt and encoding)")
        z = torch.load(src)
        tr, va = z["train"], z["val"]
    else:
        ds = load_dataset("BertilBraun/TinyPython", "small", split="train", streaming=True)
        toks = []
        for i, e in enumerate(ds):
            if i >= args.examples:
                break
            # Include instruction because code generation, not merely code continuation.
            text = PROMPT + e["task_description"] + "\n\n" + e["code"]
            toks.extend(enc.encode(text, allowed_special=set()))
            toks.append(eos)
        ids = torch.tensor(toks, dtype=torch.long)
        n = int(.9 * len(ids)); tr, va = ids[:n], ids[n:]
        torch.save({"train": tr, "val": va}, cache)

    V = enc.n_vocab
    used = torch.unique(torch.cat([tr, va]))
    inv = None
    if args.compact_vocab:
        lut = torch.full((int(used.max()) + 1,), -1, dtype=torch.long)
        lut[used] = torch.arange(used.numel())
        tr, va, V = lut[tr], lut[va], used.numel()
        inv = used            # compact id -> cl100k id, for decoding samples
    # Report unconditionally, not only on a cache miss.
    print(f"tokens {len(tr)+len(va)} train {len(tr)} val {len(va)} vocab {V} "
          f"(distinct tokens actually used: {used.numel()})")
    if not args.compact_vocab and used.numel() * 20 < V:
        print(f"  NOTE: only {used.numel()} of {V} vocab entries occur, so ~"
              f"{100*(1 - used.numel()/V):.0f}% of the embedding/output parameters and most of "
              f"the loss is vocabulary bookkeeping. Pass --compact-vocab to make the\n"
              f"        val loss reflect the layer being compared rather than the softmax size.")

    def detok(ids):
        """compact (or raw cl100k) ids -> text"""
        ids = torch.as_tensor(ids).flatten()
        if inv is not None:
            ids = inv[ids.clamp(0, inv.numel() - 1)]
        return enc.decode([int(i) for i in ids])

    return tr, va, V, detok


def get_batch(data, B, T, g, device):
    ix = torch.randint(len(data) - T - 1, (B,), generator=g)
    x = torch.stack([data[i:i + T] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + T + 1] for i in ix]).to(device)
    return x, y


class SCA2(nn.Module):
    """Embedding + one SCA2 layer (from the sca2 package) + LM head."""

    def __init__(self, V, cfg: LayerCfg, variant=None, device="cpu", layers=1,
                 ple_dim=0, tie_embed=False):
        super().__init__()
        self.e = nn.Embedding(V, cfg.d)
        # variant=None -> the version selected by SCA2_VERSION (default: latest);
        # a distinct seed per layer, else build() would make them all identical
        self.layers = nn.ModuleList([
            (build(variant, cfg, device=device, seed=i) if variant
             else make_layer(cfg, device=device, seed=i)) for i in range(layers)])
        self.layer = self.layers[0]          # back-compat for single-layer paths
        self.on = nn.LayerNorm(cfg.d)
        self.o = nn.Linear(cfg.d, V)
        # Weight tying: one V x d matrix instead of two. Standard since GPT-2 and
        # used by Llama and Mistral, and it matters here for what the loss
        # MEASURES: untied at V=32000 and d=1024 the two matrices are 65.5M
        # parameters, 46% of the smallest arm, so the comparison is diluted by a
        # softmax that no mixer touches. Tied, that drops to 30%. It removes the
        # same 32.8M from every arm, so it does not move the comparison -- it
        # stops hiding it. Off by default: a tied checkpoint has one fewer tensor
        # and will not load into an untied model, so the earlier runs stay
        # reproducible.
        self.tie_embed = tie_embed
        if tie_embed:
            self.o.weight = self.e.weight
        # PLE: per-layer embedding. One shared (V, ple_dim * layers) lookup, sliced
        # per layer and projected to d. Each layer gets its own token-identity signal
        # directly, bypassing the residual stream. 0 = off.
        self.ple_dim = ple_dim
        if ple_dim:
            self.ple_embed = nn.Embedding(V, ple_dim * layers)
            self.ple_proj = nn.ModuleList([
                nn.Linear(ple_dim, cfg.d, bias=False) for _ in range(layers)])
            # Init small so PLE starts as a perturbation, not a replacement
            for p in self.ple_proj:
                nn.init.normal_(p.weight, std=0.02)

    def core_params(self):
        return sum(p.numel() for p in self.layers.parameters())

    def forward(self, t):
        x = self.e(t)
        if self.ple_dim:
            ple = self.ple_embed(t)  # (B, T, ple_dim * L)
            L = len(self.layers)
            ple = ple.view(*ple.shape[:-1], L, self.ple_dim)  # (B, T, L, ple_dim)
        for i, layer in enumerate(self.layers):
            if self.ple_dim:
                x = x + self.ple_proj[i](ple[..., i, :])  # add before the layer
            x, _ = layer.prefill(x)
        return self.o(self.on(x))


class OriginalSCA(nn.Module):
    """The ORIGINAL SeqCond block, unmodified, from trickstr-ai/nautile-370m.

    Note what this comparison is and is not. Nautile interleaves two SeqCond
    blocks with one transformer block, so the original is a HYBRID component; the
    block carries its own SwiGLU expansion but no separate FFN, and is not
    designed to be a standalone replacement for attention. Our SCA2 layer is. So
    a single-block comparison tests the original outside its intended regime --
    it is informative about the temporal mechanism, not a verdict on Nautile.
    """

    def __init__(self, V, d=128, heads=8, anchor=2, maxlen=256, out_expand=3):
        super().__init__()
        from ref_nautile.modeling_seqcond import SeqCondBlock, RMSNorm
        self.e = nn.Embedding(V, d)
        self.blk = SeqCondBlock(d_model=d, num_heads=heads, num_query_heads=heads,
                                num_anchor_heads=anchor, num_thetas=1,
                                maxlen=maxlen, out_expand_factor=out_expand)
        self.n = RMSNorm(d)
        self.o = nn.Linear(d, V)
        self.maxlen = maxlen

    def core_params(self):
        return sum(p.numel() for p in self.blk.parameters())

    def forward(self, t):
        return self.o(self.n(self.blk(self.e(t))))


class Transformer(nn.Module):
    def __init__(self, V, d=128, heads=4, ff=256, maxlen=512, layers=1):
        super().__init__(); self.e = nn.Embedding(V, d); self.p = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, ff, dropout=0, batch_first=True,
                                       norm_first=True, activation="gelu")
            for _ in range(layers)])
        self.b = self.blocks[0]
        self.n = nn.LayerNorm(d); self.o = nn.Linear(d, V)

    def core_params(self):
        return sum(p.numel() for p in self.blocks.parameters())

    def forward(self, t):
        T = t.size(1); x = self.e(t) + self.p(torch.arange(T, device=t.device))[None]
        mask = torch.triu(torch.ones(T, T, device=t.device, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, src_mask=mask)
        return self.o(self.n(x))


@torch.no_grad()
def generate(m, prompt, n, device, temperature=0.8, top_k=40):
    """Continue `prompt` (1-D LongTensor of ids) for `n` tokens.

    SCA2 goes through its O(1) `step` path -- the same recurrence `sca2.iso`
    checks against the parallel prefill -- so this also exercises the decode
    path end to end. The Transformer recomputes the prefix, which is enough for
    a handful of short samples.
    """
    m.eval()
    x = prompt[None].to(device)
    layers = getattr(m, "layers", None)
    if layers is not None:                                  # SCA2: prefill + step
        sts = [L.init_state(1, device, torch.float32) for L in layers]
        y = m.e(x)
        for i, L in enumerate(layers):
            y, sts[i] = L.prefill(y, sts[i])
        cur = m.o(m.on(y))[:, -1]
        out = []
        for _ in range(n):
            logits = cur / max(temperature, 1e-6)
            v, ix = logits.topk(min(top_k, logits.size(-1)), -1)
            nxt = ix.gather(-1, torch.multinomial(v.softmax(-1), 1))
            out.append(int(nxt))
            hcur = m.e(nxt[:, 0])
            for i, L in enumerate(layers):
                hcur, sts[i] = L.step(hcur, sts[i])
            cur = m.o(m.on(hcur))
    else:                                                   # recompute the prefix
        ctx_max = getattr(getattr(m, "p", None), "num_embeddings", None) or \
            getattr(m, "maxlen", 512)
        out = []
        for _ in range(n):
            ctx = x[:, -ctx_max:]
            logits = m(ctx)[:, -1] / max(temperature, 1e-6)
            v, ix = logits.topk(min(top_k, logits.size(-1)), -1)
            nxt = ix.gather(-1, torch.multinomial(v.softmax(-1), 1))
            out.append(int(nxt))
            x = torch.cat([x, nxt], 1)
    m.train()
    return torch.tensor(out)


@torch.no_grad()
def evaluate(m, val, args, device):
    m.eval(); g = torch.Generator().manual_seed(999); ls = []
    for _ in range(20):
        x, y = get_batch(val, args.batch, args.block, g, device)
        ls.append(F.cross_entropy(m(x).flatten(0, 1), y.flatten()).item())
    m.train(); return sum(ls) / len(ls)


def train(name, m, tr, va, args, device):
    m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(123 + args.seed)
    npar = sum(p.numel() for p in m.parameters())
    print(name, "params", npar, flush=True)
    log = None
    if args.log:
        log = open(args.log, "a")
        log.write(json.dumps({"event": "start", "model": name, "params": npar,
                              "seed": args.seed, "lr": args.lr, "batch": args.batch,
                              "block": args.block, "steps": args.steps,
                              "vocab": int(m.o.out_features),
                              "cfg": vars(args)}) + "\n")
        log.flush()
    tok_per_step = args.batch * args.block
    train_s = 0.0                      # excludes evaluation
    for st in range(1, args.steps + 1):
        x, y = get_batch(tr, args.batch, args.block, g, device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        loss.backward(); opt.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        train_s += time.perf_counter() - t0
        if st % args.eval_every == 0 or st == args.steps or st == 1:
            # evaluation time must not be charged to tok/s -- the original clock
            # ran across evaluate(), which both deflated the number and deflated
            # it unequally between models.
            vl = evaluate(m, va, args, device)
            rec = {"event": "eval", "model": name, "seed": args.seed, "step": st,
                   "train": round(loss.item(), 5), "val": round(vl, 5),
                   "tokens": st * tok_per_step, "train_s": round(train_s, 2),
                   "tok_s": round(st * tok_per_step / train_s),
                   "epochs": round(st * tok_per_step / len(tr), 3)}
            print(name, st, "train", rec["train"], "val", rec["val"],
                  "tok/s", rec["tok_s"], "ep", rec["epochs"], flush=True)
            if log:
                log.write(json.dumps(rec) + "\n"); log.flush()
        if args.max_seconds and train_s > args.max_seconds:
            print(f"{name}: wall-clock cap reached at step {st}", flush=True)
            break
    if log:
        log.close()
    return m


def show_samples(name, m, va, a, device, detok):
    g = torch.Generator().manual_seed(7)
    print(f"\n{'='*72}\n{name}: {a.samples} continuations "
          f"({a.prompt_len}-token prompt from val, {a.sample_len} generated)\n{'='*72}")
    for k in range(a.samples):
        i = int(torch.randint(len(va) - a.prompt_len - 1, (1,), generator=g))
        prompt = va[i:i + a.prompt_len]
        out = generate(m, prompt, a.sample_len, device)
        print(f"\n--- sample {k+1} ---\n[PROMPT] {detok(prompt)!r}\n"
              f"[CONT]   {detok(out)!r}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--examples", type=int, default=20000); p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--epochs", type=float, default=0,
                   help="derive --steps from the corpus size; overrides --steps")
    p.add_argument("--block", type=int, default=128); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--d", type=int, default=128); p.add_argument("--Mc", type=int, default=64)
    p.add_argument("--Md", type=int, default=16); p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=256); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--variant", default=None,
                   help="explicit sca2 variant name; default follows SCA2_VERSION")
    p.add_argument("--freq", default="dft", help="C head positional grid: dft|len|rope")
    p.add_argument("--theta-scale", type=float, default=0.0)
    p.add_argument("--compact-vocab", action="store_true")
    p.add_argument("--only", default=None, help="sca2|transformer|original")
    p.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    p.add_argument("--log", default=None, help="append the loss curve as JSONL")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0, dest="max_seconds",
                   help="stop after this much TRAINING time (eval excluded)")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--samples", type=int, default=0,
                   help="generate this many continuations after training")
    p.add_argument("--sample-len", type=int, default=120, dest="sample_len")
    p.add_argument("--prompt-len", type=int, default=48, dest="prompt_len")
    p.add_argument("--save", default=None, help="write the trained weights here")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant = a.variant
    tr, va, V, detok = prepare(a)
    # One pass over the data is a property of the corpus, not something to
    # hand-compute from an assumed tokens-per-example -- and getting it wrong is
    # expensive: at 132 epochs this benchmark measures memorisation, with val
    # loss rising well past its minimum, so the arms are ranked on how well they
    # overfit rather than on how well they model.
    if a.epochs:
        a.steps = max(1, round(a.epochs * len(tr) / (a.batch * a.block)))
        print(f"--epochs {a.epochs} over {len(tr)} train tokens "
              f"-> {a.steps} steps of {a.batch * a.block} tokens")
    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff, freq=a.freq,
                   theta_scale=a.theta_scale, max_len=a.block)
    print(f"sca2 {variant or ('version ' + active_version())} freq={a.freq} "
          f"theta_scale={a.theta_scale} device={device}")
    def finish(name, m):
        if a.save:
            torch.save({"model": m.state_dict(), "cfg": vars(a), "V": V},
                       f"{a.save}.{name.lower()}.pt")
        if a.samples:
            show_samples(name, m, va, a, device, detok)

    if a.only not in ("transformer", "original"):
        torch.manual_seed(a.seed)
        # Name the arm after the variant. `--variant gdn_cc` goes through this
        # same wrapper (the registry returns a GDN layer, which has prefill/step
        # like any other), so logging it as "SCA2" made the two arms of a
        # SCA2-vs-GDN comparison indistinguishable in the JSONL.
        label = variant.upper() if variant else "SCA2"
        finish(label, train(label, SCA2(V, cfg, variant, device, a.layers), tr, va, a, device))
    if a.only == "original":
        torch.manual_seed(a.seed)
        finish("OriginalSCA",
               train("OriginalSCA", OriginalSCA(V, a.d, maxlen=a.block), tr, va, a, device))
    elif a.only != "sca2":
        torch.manual_seed(a.seed)
        finish("Transformer",
               train("Transformer", Transformer(V, a.d, 4, a.ff, a.block, a.layers),
                     tr, va, a, device))


if __name__ == "__main__":
    main()
