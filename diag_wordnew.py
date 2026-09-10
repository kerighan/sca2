"""Where does the word_new deficit live? Loss on word_new targets split by the
target's TRAINING-SET FREQUENCY (terciles) and by position quarter, per checkpoint.

    python diag_wordnew.py runs/ck_catch_gen3_s0.*.pt runs/ck_catch_gdn_s0.*.pt [...]

If GDN's edge on word_new is on FREQUENT tokens (common identifiers like ' np',
' os', ' data'), the missing piece is a language prior -- embedding/FFN/read-out
path, not memory. If it is on RARE tokens, it is context: something in the window
short of an exact repeat (a stem, a sibling identifier) that a similarity memory
exploits and a hash cannot.
"""
import sys, torch, torch.nn.functional as F
from sca2 import tokclass as tc
from diag_tokclass import load
from pretrain import batches

paths = sys.argv[1:]
dev = "cuda"
tab = tc.load_table("pycode_bpe16k")
z = torch.load("pycode_long1024.pt"); tr, val = z["train"], z["val"]
freq = torch.bincount(tr[: 100_000_000].long(), minlength=16384).float()
r = freq.argsort(descending=True).argsort()          # rank of each id, 0 = most frequent
Q = 4; NB = 60; B = 8; T = 1024
bounds = [0, 300, 1500, 16384]                      # rank buckets: top-300 / 300-1500 / tail
for p in paths:
    m, cfg = load(p, dev)
    S = torch.zeros(3, Q, device=dev); N = torch.zeros(3, Q, device=dev)
    S_all = torch.zeros(3, device=dev); N_all = torch.zeros(3, device=dev)
    with torch.no_grad():
        for k, (x, y) in enumerate(batches(val, B, T, dev)):
            if k >= NB: break
            lo = F.cross_entropy(m(x).flatten(0, 1), y.flatten(), reduction="none").view(B, T)
            cls = tc.split_repeat(x, y, tab)
            rk = r.to(dev)[y]
            pos = (torch.arange(T, device=dev) * Q // T)[None].expand(B, T)
            new = cls == tc.WORD_NEW
            for i in range(3):
                sel = new & (rk >= bounds[i]) & (rk < bounds[i + 1])
                S_all[i] += lo[sel].sum(); N_all[i] += sel.sum()
                for q in range(Q):
                    s2 = sel & (pos == q); S[i, q] += lo[s2].sum(); N[i, q] += s2.sum()
    name = p.split("/")[-1].split(".")[0]
    print(f"\n{name}: word_new loss by target frequency rank (share of word_new tokens)")
    for i, lab in enumerate(("top-300", "300-1500", "tail")):
        sh = (N_all[i] / N_all.sum()).item()
        print(f"  {lab:>9} {sh:5.1%}  all {S_all[i]/N_all[i]:.4f}   by pos quarter: "
              + " ".join(f"{(S[i,q]/N[i,q]).item():.3f}" for q in range(Q)))
    del m; torch.cuda.empty_cache()
