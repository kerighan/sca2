"""
Synthetic copy: can the layer reproduce a random string it has just read?

WHY THIS TASK. Every comparison run so far is language modelling, where a good
short-range prior gets you most of the loss and a model can look competent while
remembering almost nothing verbatim. Copying admits no prior: the string is
freshly random each batch, so the ONLY way to emit token i of the copy is to have
carried it, exactly, in the state across the separator. That makes this a direct
read on state capacity and on whether the addressing mechanism can retrieve by
position rather than by content-similarity.

    [BOS] s_1 .. s_L [SEP] s_1 .. s_L [PAD ...]

Loss and accuracy are scored ONLY on the second copy. Lengths are mixed during
training (a curriculum-free uniform draw over --lengths) and reported separately
at eval, so the output is a capacity CURVE, not one number: accuracy stays near 1
up to the point where the state runs out, then falls off. That cliff is the
quantity of interest.

WHAT IS AND IS NOT A FAIR COMPARISON HERE. A recurrent layer carries a state of
fixed size; the transformer carries the whole prefix, so it has O(T) memory and
should win outright -- it is included as a CEILING, not as a peer. Among the
recurrent arms the honest axis is accuracy against state size, which is why the
state footprint (floats per sequence, read off init_state) is printed next to
each arm. An arm that wins while carrying 4x the state has not won.

Padding is trailing, so it costs the recurrent arms nothing that matters: those
positions are masked out of the loss and the state is never read after them.

    python bench_copy.py --arms v3polarflat_cc,gdn_cc,transformer --steps 4000
    python bench_copy.py --arms v3polarflat_cc,gdn_cc --Md 4,16   # capacity sweep
"""
import argparse, json, time
import torch
import torch.nn.functional as F

from sca2.ref import LayerCfg
from bench_tinypython import SCA2, Transformer


def make_batch(B, L, S, Tmax, device, g):
    """(x, y) for one batch of copy problems, all of the same length L.

    y is -100 everywhere except the positions whose NEXT token is a copy token,
    i.e. 1+L .. 2L inclusive: position 1+L sees [SEP] and must emit s_1, and
    position 2L sees s_{L-1} and must emit s_L.
    """
    BOS, SEP, PAD = S, S + 1, S + 2
    s = torch.randint(0, S, (B, L), generator=g)
    x = torch.full((B, Tmax), PAD, dtype=torch.long)
    x[:, 0] = BOS
    x[:, 1:1 + L] = s
    x[:, 1 + L] = SEP
    x[:, 2 + L:2 + 2 * L] = s
    y = torch.full((B, Tmax), -100, dtype=torch.long)
    y[:, 1 + L:1 + 2 * L] = s
    return x.to(device), y.to(device)


def loss_of(m, x, y):
    return F.cross_entropy(m(x).flatten(0, 1), y.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate(m, lengths, B, S, Tmax, device, nb=8, seed=1234):
    """Per-length token accuracy and whole-string exact-match rate."""
    m.eval()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for L in lengths:
        tok, exact, n = 0, 0, 0
        for _ in range(nb):
            x, y = make_batch(B, L, S, Tmax, device, g)
            pred = m(x).argmax(-1)
            sl = slice(1 + L, 1 + 2 * L)
            hit = pred[:, sl] == y[:, sl]
            tok += hit.sum().item()
            exact += hit.all(-1).sum().item()
            n += hit.numel()
        out[L] = (tok / n, exact / (nb * B))
    m.train()
    return out


def _numels(st):
    """Total floats in a state, recursing into the per-head sub-dicts."""
    if torch.is_tensor(st):
        return st.numel()
    if isinstance(st, dict):
        return sum(_numels(v) for v in st.values())
    return 0


def state_floats(m):
    """Floats the recurrent state carries across the separator, per sequence.

    None for the transformer: its 'state' is the KV cache, which grows with T
    and so is not comparable to a fixed-size recurrent state.

    The first attempt at this summed only the TOP-LEVEL tensors of the state
    dict, which for SCA2 is just z_prev -- it reported 256 floats against GDN's
    24840 and made the task look rigged. SCA2's state lives one level down, in
    the 'c' and 'd' sub-dicts, and actually totals MORE than GDN's.
    """
    layers = getattr(m, "layers", [])
    if not len(layers) or not hasattr(layers[0], "init_state"):
        return None
    return _numels(layers[0].init_state(1, "cpu", torch.float32)) * len(layers)


def parse_arm(spec, a):
    """'v3polarflat_cc/Mc=32:Md=8' -> (variant, dict of overrides).

    Sweeping a capacity knob is the whole point of this file, and the knob that
    matters for SCA2 is Mc: the C head state is 2.Mc.dv floats (16384 at Mc=128,
    d=128) against the D head's 2.Md.dv (512 at Md=4), so ~97% of what the layer
    carries is indexed by Mc. One process per value would re-pay compilation and
    give every arm a different data order, so overrides are per-arm instead.

    Overrides are separated by ':', NOT by ',' -- ',' already separates arms, so
    'layers=4,Mc=64' silently parsed as two arms, one of them named 'Mc=64'.
    """
    name, _, rest = spec.partition("/")
    if "," in rest:
        raise SystemExit(f"separate overrides with ':' not ',': {spec!r}")
    ov = {}
    for kv in filter(None, rest.split(":")):
        k, _, v = kv.partition("=")
        if not hasattr(a, k):
            raise SystemExit(f"unknown override {k!r} in {spec!r}")
        ov[k] = type(getattr(a, k))(v)
    return name, ov


def build_arm(spec, V, a, Tmax, device):
    name, ov = parse_arm(spec, a)

    def cf(k):
        return ov.get(k, getattr(a, k))

    torch.manual_seed(0)
    if name == "transformer":
        m = Transformer(V, cf("d"), 4, cf("trf_ff"), Tmax, cf("layers")).to(device)
        blocks = m.blocks
        if not a.no_compile:
            m = torch.compile(m, dynamic=False)
            m.core_params = lambda: sum(p.numel() for p in blocks.parameters())
            m.layers = []
        return m
    cfg = LayerCfg(cf("d"), cf("Mc"), cf("Md"), cf("G"), cf("ff"), freq=cf("freq"),
                   theta_scale=cf("theta_scale"), max_len=Tmax,
                   dv=cf("dv") or None, Ls=cf("Ls"),
                   gdn_heads=cf("gdn_heads"), gdn_head_k=cf("gdn_head_k"),
                   gdn_expand_v=cf("gdn_expand_v"))
    return SCA2(V, cfg, name, device, cf("layers")).to(device)


def run(arm, a, lengths, Tmax, device, log):
    S = a.symbols
    V = S + 3
    m = build_arm(arm, V, a, Tmax, device)
    nl = len(m.layers) or a.layers
    sf = state_floats(m)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    g = torch.Generator().manual_seed(0)
    print(f"\n=== {arm}  params/layer {m.core_params() // nl}  "
          f"state {sf if sf is not None else 'O(T) kv'} floats ===", flush=True)

    # warmup outside the clock: compilation, and one batch per length so every
    # shape is already traced when the timed section starts
    for L in lengths:
        x, y = make_batch(a.batch, L, S, Tmax, device, g)
        opt.zero_grad(set_to_none=True); loss_of(m, x, y).backward(); opt.step()
    torch.cuda.synchronize()

    hard = max(lengths)
    solved, hits = None, 0
    t0 = time.time()
    for step in range(1, a.steps + 1):
        L = lengths[int(torch.randint(len(lengths), (1,), generator=g))]
        x, y = make_batch(a.batch, L, S, Tmax, device, g)
        opt.zero_grad(set_to_none=True)
        loss = loss_of(m, x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            torch.cuda.synchronize()
            el = time.time() - t0
            acc = evaluate(m, lengths, a.eval_batch, S, Tmax, device, a.eval_batches)
            row = {"arm": arm, "step": step, "s": round(el, 1),
                   "train": round(loss.item(), 4), "state": sf,
                   "acc": {L: round(v[0], 4) for L, v in acc.items()},
                   "exact": {L: round(v[1], 4) for L, v in acc.items()}}
            log.write(json.dumps(row) + "\n"); log.flush()
            print(f"  step {step:6d} {el:6.0f}s  train {loss.item():.4f}  " +
                  "  ".join(f"L{L}:{v[0]:.2f}/{v[1]:.2f}" for L, v in acc.items()),
                  flush=True)
            # Stop once the hardest length is solved, and require two evals in a
            # row: exact-match on a 64-sequence eval swings by a few points
            # between evals even after the curve has flattened, so a single
            # crossing would report a step count that is mostly noise.
            hits = hits + 1 if acc[hard][1] >= a.target else 0
            if hits >= 2:
                solved = step
                print(f"  solved L{hard} at step {step} ({el:.0f}s)", flush=True)
                break
    final = {L: v[1] for L, v in acc.items()}
    del m
    torch.cuda.empty_cache()
    return {"arm": arm, "state": sf, "solved": solved, "steps": step,
            "s": round(time.time() - t0, 1), "exact": final}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="v3polarflat_cc,gdn_cc,transformer")
    p.add_argument("--lengths", default="8,16,32,64,128")
    # Tmax normally follows the longest length. Pinning it decouples the two
    # things that changed between the first two runs of this file: the padded
    # sequence length, and whether an out-of-capacity length sits in the
    # training mix. Only one of those can be the reason the first run failed.
    p.add_argument("--tmax", type=int, default=0,
                   help="pad to this length instead of 2*max(lengths)+2")
    p.add_argument("--symbols", type=int, default=64,
                   help="alphabet size; entropy per token is log2(S) bits")
    p.add_argument("--steps", type=int, default=4000, help="budget, not a target")
    # Early stop: the useful number is STEPS TO SOLVE, not accuracy at a fixed
    # step. A fixed budget spends the same wall clock on an arm that solved the
    # task in 2k steps as on one that never will, and -- as the first run of
    # this file showed -- reports an arm as capacity-limited when it was merely
    # undertrained. Set --target 2 to disable and always run the full budget.
    p.add_argument("--target", type=float, default=0.95,
                   help="exact-match at the longest length that counts as solved")
    p.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    p.add_argument("--eval-batches", type=int, default=4, dest="eval_batches")
    p.add_argument("--eval-batch", type=int, default=64, dest="eval_batch")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--Mc", type=int, default=128)
    p.add_argument("--Md", type=int, default=4)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--ff", type=int, default=364)
    p.add_argument("--trf-ff", type=int, default=364, dest="trf_ff")
    # GDN's state is gdn_heads.head_k^2.expand_v; defaults are the param-matched
    # shape. Override per arm, e.g. gdn_cc/gdn_head_k=84, to put GDN and SCA2 on
    # the same state-size axis.
    # LapA knobs: value width (0 -> d//2), short-head window, content-phase init.
    # theta_scale used to be hardcoded to 0.0 here; the champion trains at 0.02.
    p.add_argument("--dv", type=int, default=0)
    p.add_argument("--Ls", type=int, default=16)
    p.add_argument("--theta-scale", type=float, default=0.02, dest="theta_scale")
    p.add_argument("--gdn-heads", type=int, default=3, dest="gdn_heads")
    p.add_argument("--gdn-head-k", type=int, default=60, dest="gdn_head_k")
    p.add_argument("--gdn-expand-v", type=float, default=1.0, dest="gdn_expand_v")
    p.add_argument("--freq", default="rope")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--log", default="runs/copy.jsonl")
    p.add_argument("--no-compile", action="store_true", dest="no_compile")
    a = p.parse_args(argv)
    device = "cuda"

    lengths = [int(v) for v in a.lengths.split(",")]
    Tmax = a.tmax or 2 * max(lengths) + 2
    if Tmax < 2 * max(lengths) + 2:
        raise SystemExit(f"--tmax {Tmax} too small for length {max(lengths)}")
    print(f"copy task: alphabet {a.symbols}, lengths {lengths}, Tmax {Tmax}, "
          f"B={a.batch}, {a.steps} steps/arm")
    print("reported per length: token accuracy / exact-string rate\n")

    log = open(a.log, "a")
    res = [run(arm, a, lengths, Tmax, device, log) for arm in a.arms.split(",")]
    log.close()

    print(f"\n{'arm':28s} {'state':>8s} {'solve':>7s} {'s':>6s}   "
          + " ".join(f"L{L:<5d}" for L in lengths))
    for r in res:
        st = r["state"] if r["state"] is not None else "kv"
        sv = r["solved"] if r["solved"] else f">{r['steps']}"
        print(f"{r['arm']:28s} {str(st):>8s} {str(sv):>7s} {r['s']:6.0f}   "
              + " ".join(f"{r['exact'][L]:<6.2f}" for L in lengths))
    # returned, not just printed, so a driver can decide what to run next --
    # bench_lmax.py walks a length ladder and needs to know where it stopped
    return res


if __name__ == "__main__":
    raise SystemExit(main())
