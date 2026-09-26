"""Generate text from a trained checkpoint, and cost one decoded token.

    python decode.py --ckpt runs/ck_zyda.z_dv256.pt --samples 3
    python decode.py --ckpt runs/ck_zyda.z_dv256.pt runs/ck_zyda.z_gdn.pt --bench

Both arms are recurrent, so both decode in O(1) per token: neither re-reads the
prefix. What differs is the CONSTANT, and the constant is set by how many bytes
have to move per token -- the weights, plus the recurrent state.

Read the two halves of --bench differently.

The bytes are architecture. They are what a deployed decoder at batch 1, which
is memory-bandwidth bound, actually pays, and they do not depend on anyone's
kernel.

The milliseconds are NOT architecture, and are reported only for
implementation parity. Neither step path is optimised: LapA's is plain PyTorch
and so is the GDN baseline's, which calls fla's `naive_recurrent_gated_delta_
rule`. fla also ships `fused_recurrent_gated_delta_rule`, a Triton kernel we
have no counterpart for, so a benchmark that used it would measure the kernel
and not the model. Comparing the two naive paths keeps the implementations at
the same level, which is the only way the number says anything -- and it still
says more about Python overhead than about either design.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from bench_tinypython import SCA2, generate
from sca2.ref import LayerCfg

# Only the fields LayerCfg accepts; the checkpoint's cfg is the whole argparse
# namespace and carries many more.
CFG_FIELDS = None


def load(path: str, device: str):
    global CFG_FIELDS
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg_d, V = ck["cfg"], ck["V"]
    if CFG_FIELDS is None:
        import dataclasses
        CFG_FIELDS = {f.name for f in dataclasses.fields(LayerCfg)}
    kw = {k: v for k, v in cfg_d.items() if k in CFG_FIELDS}
    kw["d"] = cfg_d["d"]
    kw["max_len"] = cfg_d["block"]
    if "Mc" in cfg_d:
        kw["Mc"] = cfg_d["Mc"]
    if "damp_mem" in cfg_d and isinstance(cfg_d["damp_mem"], str):
        lo, hi = cfg_d["damp_mem"].split(",")
        kw["damp_mem"] = (float(lo), float(hi))
    cfg = LayerCfg(**kw)
    torch.manual_seed(0)
    m = SCA2(V, cfg, cfg_d["variant"], device, cfg_d["layers"],
             tie_embed=cfg_d.get("tie_embed", False)).to(device)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    # `layer.*` is the single-layer back-compat alias of `layers.0.*`.
    #
    # lam_anchor_mask/raw are registered unconditionally but only read when
    # cfg.lam_anchor > 0, and both are rebuilt exactly from the seeded init
    # rather than learned -- so a checkpoint written before the flag existed is
    # complete, and reconstructing them is identity, not a guess. With anchors
    # ON they carry which modes are pinned, and missing is then a real error.
    inert = () if cfg.lam_anchor else ("lam_anchor_mask", "lam_anchor_raw")
    real_missing = [k for k in missing
                    if not k.startswith("layer.")
                    and not k.rsplit(".", 1)[-1] in inert]
    if real_missing:
        raise SystemExit(f"{path}: missing weights {real_missing[:6]}")
    m.eval()
    return m, cfg, cfg_d, V


def state_bytes(m, cfg, device) -> int:
    """Bytes of recurrent state one sequence carries between tokens."""
    total = 0
    for layer in m.layers:
        st = layer.init_state(1, device, torch.float32)

        def walk(o):
            nonlocal total
            if torch.is_tensor(o):
                total += o.numel() * o.element_size()
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v)
        walk(st)
    return total


PROMPTS = [
    "The main difference between a recurrent model and a transformer is",
    "In 1892, the city council decided to",
    "def compute_loss(model, batch):",
    "The recipe calls for three cups of",
    "According to the report published last week,",
    "She opened the door and saw",
    "The patient presented with a persistent",
    "import numpy as np\n\ndef",
    "Q: What is the capital of Australia?\nA:",
    "1. First, gather the materials.\n2.",
]


def rep_metrics(ids: list[int], n: int = 4):
    """(rep-n, distinct-3, distinct-1, H1, top10) over ONE continuation.

    rep-n is the fraction of n-grams that occur more than once: 0 for text that
    never repeats itself, approaching 1 for a loop. distinct-3 is unique
    trigrams over total, so it falls as the text degenerates. Both read the
    token ids, not the text, so tokenisation cannot flatter either arm.
    """
    def grams(k):
        return [tuple(ids[i:i + k]) for i in range(len(ids) - k + 1)]
    g = grams(n)
    if not g:
        return 0.0, 1.0, 1.0, 0.0, 1.0
    from collections import Counter
    import math as _m
    c = Counter(g)
    repeated = sum(v for v in c.values() if v > 1)
    g3 = grams(3)
    # VOCABULARY, not n-grams. A model can repeat few 4-grams while recycling
    # the same twenty words forever; distinct-1 and the unigram entropy see
    # that, rep-4 does not.
    u = Counter(ids)
    tot = len(ids)
    h1 = -sum((v / tot) * _m.log(v / tot) for v in u.values())
    top10 = sum(v for _, v in u.most_common(10)) / tot
    return (repeated / len(g), len(set(g3)) / max(len(g3), 1),
            len(u) / tot, h1, top10)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--bpe", default="zyda_bpe32k")
    p.add_argument("--samples", type=int, default=0)
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--prompt", default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40, dest="top_k")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--rep", type=int, default=0,
                   help="N seeds per prompt: measure degenerate repetition, which "
                        "the loss averages over and three samples cannot settle")
    p.add_argument("--bench-tokens", type=int, default=64, dest="bench_tokens")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    from tokenizers import ByteLevelBPETokenizer
    tk = ByteLevelBPETokenizer(f"{a.bpe}-vocab.json", f"{a.bpe}-merges.txt")

    rows = []
    for path in a.ckpt:
        m, cfg, cfg_d, V = load(path, a.device)
        name = cfg_d.get("label", path)
        nparam = sum(p_.numel() for p_ in m.parameters())
        sb = state_bytes(m, cfg, a.device)

        if a.samples:
            print(f"\n{'='*74}\n{name}   ({nparam/1e6:.1f}M params)\n{'='*74}")
            prompts = ([a.prompt] * a.samples if a.prompt else [
                "The main difference between a recurrent model and a transformer is",
                "In 1892, the city council decided to",
                "def compute_loss(model, batch):",
            ][:a.samples])
            for k, text in enumerate(prompts):
                ids = torch.tensor(tk.encode(text).ids, dtype=torch.long)
                out = generate(m, ids, a.tokens, a.device,
                               temperature=a.temperature, top_k=a.top_k)
                print(f"\n--- {k+1} ---\n[PROMPT] {text}\n"
                      f"[CONT]   {tk.decode(out.tolist())}", flush=True)

        if a.rep:
            import statistics
            from collections import Counter
            reps, dis, worst = [], [], (0.0, "")
            d1s, h1s, t10s = [], [], []
            for pi, text in enumerate(PROMPTS):
                ids = torch.tensor(tk.encode(text).ids, dtype=torch.long)
                for sd in range(a.rep):
                    # PAIRED: the same (prompt, seed) is used for every arm, so a
                    # difference is the model and not the sampler.
                    torch.manual_seed(1000 * pi + sd)
                    out = generate(m, ids, a.tokens, a.device,
                                   temperature=a.temperature, top_k=a.top_k)
                    r, d, d1, h1, t10 = rep_metrics(out.tolist())
                    reps.append(r)
                    dis.append(d)
                    d1s.append(d1); h1s.append(h1); t10s.append(t10)
                    if r > worst[0]:
                        worst = (r, tk.decode(out.tolist())[:90])
            print(f"\n{name}  ({len(reps)} generations of {a.tokens} tokens, "
                  f"{len(PROMPTS)} prompts x {a.rep} seeds)")
            print(f"  rep-4      median {statistics.median(reps):.3f}  "
                  f"mean {statistics.mean(reps):.3f}  "
                  f"frac > 0.5 {sum(r > 0.5 for r in reps)/len(reps):.2f}")
            print(f"  distinct-3 median {statistics.median(dis):.3f}  "
                  f"mean {statistics.mean(dis):.3f}")
            print(f"  distinct-1 median {statistics.median(d1s):.3f}   "
                  f"unigram H {statistics.median(h1s):.3f} nats   "
                  f"top-10 share {statistics.median(t10s):.3f}")
            print(f"  worst      rep-4 {worst[0]:.3f}  {worst[1]!r}")

        if a.bench:
            x = torch.randint(0, V, (1, 256), device=a.device)
            with torch.no_grad():
                sts = [L.init_state(1, a.device, torch.float32) for L in m.layers]
                y = m.e(x)
                for i, L in enumerate(m.layers):
                    y, sts[i] = L.prefill(y, sts[i])
                tok = x[:, -1]
                for _ in range(8):                          # warm the caches
                    h = m.e(tok)
                    s2 = [s for s in sts]
                    for i, L in enumerate(m.layers):
                        h, s2[i] = L.step(h, s2[i])
                torch.cuda.synchronize()
                per = []
                for _ in range(5):
                    s2 = [s for s in sts]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(a.bench_tokens):
                        h = m.e(tok)
                        for i, L in enumerate(m.layers):
                            h, s2[i] = L.step(h, s2[i])
                        _ = m.o(m.on(h))
                    torch.cuda.synchronize()
                    per.append((time.perf_counter() - t0) / a.bench_tokens)
            ms = statistics.median(per) * 1e3
            rows.append((name, nparam, sb, ms))
        del m
        torch.cuda.empty_cache()

    if rows:
        print(f"\n{'='*74}\ndecode cost, batch 1")
        print("bytes = architecture; ms = naive PyTorch on both sides, "
              "implementation parity only")
        print(f"{'arm':>10} {'params':>12} {'weights':>10} {'state':>10} "
              f"{'total/token':>12} {'ms/token':>10} {'tok/s':>9}")
        for name, n, sb, ms in rows:
            wb = n * 2                                      # bf16 serving
            print(f"{name:>10} {n:12,} {wb/2**20:9.1f}M {sb/2**20:9.2f}M "
                  f"{(wb+sb)/2**20:11.1f}M {ms:10.2f} {1e3/ms:9.1f}")
        base = rows[-1]
        print(f"\nrelative to {base[0]}:")
        for name, n, sb, ms in rows[:-1]:
            print(f"  {name:>10}  bytes x{(n*2+sb)/(base[1]*2+base[2]):.3f}  "
                  f"ms x{ms/base[3]:.3f}")


if __name__ == "__main__":
    main()
