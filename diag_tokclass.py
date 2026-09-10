"""Loss by TOKEN CLASS on saved checkpoints -- the zero-GPU-hour test of
CATCHUP.md conjecture 1 (hash keys vs metric keys).

    python diag_tokclass.py runs/ck_A.pt runs/ck_B.pt [...]  [--nb 60] [--buckets 2]

Prints, per checkpoint, the val loss on each class of sca2/tokclass.py and, with
two or more checkpoints, the gap of every later one against the FIRST. --buckets
splits each class further by position in the window (early/late halves by
default) so "word_rep late in the window" -- the C head's home turf -- is read
on its own.

What each column means for the conjecture, with A = generation 3 and B = GDN:
    word_rep   exact repeat of a token already in the window: retrieval. The
               torus code should hold its lead here, and lose it last.
    word_new   first occurrence in the window: nothing to retrieve, only
               similarity to other contexts helps. If GDN's catch-up lives
               here, the metric-keys reading is supported and cdelta_bp is the
               fix to try; if it lives in word_rep, it is not.
    punct/ws/kw  local syntax; where the short-range gap closed first.

Checkpoints are in pretrain.py --save format ({"model","cfg","V"}); the model is
rebuilt from the saved cfg, so shapes and variant need not be given. Eval is on
the val split of cfg["data"], the same windows pretrain.py used.
"""
import argparse
import sys
import torch
import torch.nn.functional as F

from sca2.ref import LayerCfg
from sca2 import tokclass as tc
from bench_tinypython import SCA2
from pretrain import batches


def load(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["cfg"]
    g = lambda k, d=None: a.get(k, d)
    cfg = LayerCfg(a["d"], a["Mc"], a["Md"], a["G"], a["ff"], freq=g("freq", "rope"),
                   theta_scale=g("theta_scale", 0.0), max_len=a["block"], dv=g("dv"),
                   c_heads=g("c_heads", 1), c_decay=g("c_decay", False),
                   c_decay_init=g("c_decay_init", "gdn"), c_sepq=g("c_sepq", False),
                   conv=g("conv", 0), gated_read=g("gated_read", False),
                   gdn_heads=g("gdn_heads", 3), gdn_head_k=g("gdn_head_k", 60),
                   gdn_expand_v=g("gdn_expand_v", 1.0))
    # the uncompiled variant: same parameters, no compile step for a one-off eval
    variant = a["variant"][:-3] if a["variant"].endswith("_cc") else a["variant"]
    m = SCA2(ck["V"], cfg, variant, device, a["layers"])
    # a _cc checkpoint was saved through the compile wrapper, whose module sits
    # at layers.<i>.layer.*; SCA2 also aliases layers.0 as .layer, saved twice
    sd = {}
    for k, v in ck["model"].items():
        parts = k.split(".")
        if parts[0] == "layer" and len(parts) > 1 and parts[1] == "layer":
            del parts[1]                    # alias of layers.0, same tensors
        elif parts[0] == "layers" and len(parts) > 2 and parts[2] == "layer":
            del parts[2]
        sd[".".join(parts)] = v
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval(), a


@torch.no_grad()
def profile(m, val, B, T, device, tab, nb, buckets):
    K = len(tc.NAMES)
    S = torch.zeros(buckets, K, device=device)
    N = torch.zeros(buckets, K, device=device)
    tot, n = 0.0, 0
    w = T // buckets
    for k, (x, y) in enumerate(batches(val, B, T, device)):
        if k >= nb:
            break
        lo = F.cross_entropy(m(x).flatten(0, 1), y.flatten(), reduction="none").view(B, T)
        cls = tc.split_repeat(x, y, tab)
        tot += lo.mean().item(); n += 1
        for b in range(buckets):
            sl = slice(b * w, (b + 1) * w)
            s_, n_ = tc.class_means(lo[:, sl], cls[:, sl])
            S[b] += s_; N[b] += n_
    return tot / n, (S / N.clamp(min=1)).cpu(), (N.sum(0) / N.sum()).cpu()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--nb", type=int, default=60, help="eval batches (pretrain.py used 60)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--buckets", type=int, default=2, help="position halves per class")
    p.add_argument("--bpe", default="pycode_bpe16k")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(argv)
    tab = tc.load_table(a.bpe)

    res = []
    for path in a.ckpts:
        m, cfg = load(path, a.device)
        z = torch.load(cfg["data"])
        val = z["val"]
        T = cfg["block"]
        v, P, share = profile(m, val, a.batch, T, a.device, tab, a.nb, a.buckets)
        res.append((path, cfg, v, P, share))
        print(f"{path}: {cfg['variant']} L{cfg['layers']} Mc{cfg['Mc']} dv{cfg.get('dv')} "
              f"seed {cfg.get('seed', 0)}   val {v:.4f}", flush=True)
        del m; torch.cuda.empty_cache()

    share = res[0][4]
    K = len(tc.NAMES)
    print(f"\nshare of tokens: " + "  ".join(f"{tc.NAMES[i]} {share[i]:.1%}" for i in range(K)))
    hdr = f"{'ckpt':>28} {'bucket':>7} " + " ".join(f"{n:>9}" for n in tc.NAMES)
    print("\n" + hdr)
    for path, cfg, v, P, _ in res:
        for b in range(a.buckets):
            lab = f"{b}/{a.buckets}"
            print(f"{path[-28:]:>28} {lab:>7} " + " ".join(f"{P[b, i]:9.4f}" for i in range(K)))
    if len(res) > 1:
        print("\nGAP vs first checkpoint (negative = later ckpt better)")
        print(hdr)
        P0 = res[0][3]
        for path, cfg, v, P, _ in res[1:]:
            for b in range(a.buckets):
                lab = f"{b}/{a.buckets}"
                print(f"{path[-28:]:>28} {lab:>7} "
                      + " ".join(f"{(P[b, i] - P0[b, i]):+9.4f}" for i in range(K)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
