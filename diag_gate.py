"""Does the key-verification gate DISCRIMINATE? Mean gate g and cosine evidence m per
token class, per layer, on a cdelta_kv / cshort_kv checkpoint. Independent of the loss.

    python diag_gate.py runs/ck_catch_shortkv_s0.*.pt [--nb 20]

Expected if the mechanism works: g high on word_rep (the memory found its own key
back), low on word_new (nothing to find, the key read is a mixture). If g is flat
or the slope a has collapsed to 0, the gate is a passenger.
"""
import argparse, sys, torch, torch.nn.functional as F
from sca2 import tokclass as tc
from sca2.arch_cdelta import CHeadDeltaKV
from diag_tokclass import load
from pretrain import batches

p = argparse.ArgumentParser(); p.add_argument("ckpt"); p.add_argument("--nb", type=int, default=20)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
m, cfg = load(a.ckpt, dev)
val = torch.load(cfg["data"])["val"]; tab = tc.load_table("pycode_bpe16k")
REC = []
orig = CHeadDeltaKV._out
def spy(self, u, z):
    dv, dk = self.dv_out, self.DK
    re = u[..., :dv + dk]
    mm = F.cosine_similarity(re[..., dv:], self.Kv(z), dim=-1, eps=1e-6)
    g = torch.sigmoid(self.ga * mm + self.gb)
    REC.append((mm.detach(), g.detach()))
    return orig(self, u, z)
CHeadDeltaKV._out = spy
B, T = 8, cfg["block"]
S = {}
with torch.no_grad():
    for k, (x, y) in enumerate(batches(val, B, T, dev)):
        if k >= a.nb: break
        REC.clear(); m(x)
        cls = tc.split_repeat(x, y, tab); keep = (torch.arange(T, device=dev) >= 64)[None].expand(B, T)
        groups = {"word_new": cls == tc.WORD_NEW, "word_rep": cls == tc.WORD_REP,
                  "other": (cls != tc.WORD_NEW) & (cls != tc.WORD_REP)}
        # one record per CHUNK per layer, in order (layer 0's chunks, then layer 1's, ...):
        # concatenate per layer. Damped heads overflow float32 if CTX*lambda > ~60,
        # so run this with SCA2_CTX_CHUNK <= 256, never 1024.
        n_layers = len(m.layers); per = len(REC) // n_layers
        REC[:] = [(torch.cat([r[0] for r in REC[i*per:(i+1)*per]], 1),
                   torch.cat([r[1] for r in REC[i*per:(i+1)*per]], 1)) for i in range(n_layers)]
        for L, (mm, g) in enumerate(REC):
            for name, sel in groups.items():
                sel = sel & keep
                S.setdefault((L, name), []).append(torch.stack([mm[sel], g[sel]], 1).cpu())
layers = sorted({L for L, _ in S})
heads = [getattr(l, "layer", l).c for l in m.layers]
print(f"{a.ckpt.split('/')[-1]}: gate slope a and bias b per layer: "
      + "  ".join(f"L{L}: a={heads[L].ga.item():+.2f} b={heads[L].gb.item():+.2f}" for L in layers))
print(f"\n{'layer':>5} {'class':>9} {'mean cos m':>11} {'mean gate g':>12} {'P(g<0.25)':>10} {'P(g>0.75)':>10}")
for L in layers:
    for name in ("word_rep", "word_new", "other"):
        v = torch.cat(S[(L, name)]); mm, g = v[:, 0], v[:, 1]
        print(f"{L:>5} {name:>9} {mm.mean():11.3f} {g.mean():12.3f} {(g<0.25).float().mean():10.3f} {(g>0.75).float().mean():10.3f}")
CHeadDeltaKV._out = orig
