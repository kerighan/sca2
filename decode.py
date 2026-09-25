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
    real_missing = [k for k in missing if not k.startswith("layer.")]
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
