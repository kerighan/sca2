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
import argparse, json, math, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from sca2.ref import LayerCfg
from bench_tinypython import SCA2, Transformer, generate

CLS_TAB = None      # set by --class-eval; see evaluate()
AMP = None          # set by --amp; None = float32, the whole d=128 campaign's setting


def amp_ctx():
    """Autocast context for the forward pass. --amp bf16 applies to EVERY arm.

    The d=128 campaign trained in float32 throughout, so that stays the default and
    nothing about those runs moves. At d=1024 on GB10 float32 costs ~2.2x (SPARK.md
    §9) and the machine has bf16 tensor cores, so the Spark runs pass --amp bf16.
    The loss is always taken on float32 logits, and the layer keeps its recurrent
    state and phases in float32 regardless (lapa/layer.py precision policy)."""
    return torch.autocast("cuda", dtype=AMP, enabled=AMP is not None)


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
def evaluate(m, val, B, T, device, nb=25, buckets=0, cls_tab=None):
    """Mean val loss, and optionally its breakdown by POSITION in the window.

    The aggregate loss cannot settle an architecture comparison here: its
    standard deviation across evals is 0.039 nats at nb=25 (measured on the four
    arms of runs/h4.jsonl in a window where the curve is nearly flat), which is
    the size of every effect looked for so far. The per-position profile is a
    WITHIN-MODEL relative measure -- loss late in the window against loss early
    in it -- so a model's own noise largely cancels, and it answers the actual
    question about long context: does the extra distance get used?

    The last position is excluded: with a flat token stream its target is the
    first token of the next window (see prep_longdoc.py).

    cls_tab (sca2.tokclass.load_table) adds a breakdown by WHAT the target token
    is -- word seen earlier in the window / new word / keyword / punctuation ...
    -- the instrument for CATCHUP.md conjecture 1. Same forward, so it is free.
    Returns (val, pos_profile, {class: loss} or None).
    """
    m.eval()
    ls, bk, nb_seen = [], None, 0
    cs = cn = None
    w = 0 if not buckets else (T - 1) // buckets
    for k, (x, y) in enumerate(batches(val, B, T, device)):
        if k >= nb:
            break
        with amp_ctx():
            lg = m(x)
        lo = F.cross_entropy(lg.float().flatten(0, 1), y.flatten(),
                             reduction="none").view(B, T)
        ls.append(lo.mean().item())
        if w:
            b = lo[:, :buckets * w].view(B, buckets, w).mean((0, 2))
            bk = b if bk is None else bk + b
            nb_seen += 1
        if cls_tab is not None:
            from sca2 import tokclass as tc
            s_, n_ = tc.class_means(lo, tc.split_repeat(x, y, cls_tab))
            cs = s_ if cs is None else cs + s_
            cn = n_ if cn is None else cn + n_
    m.train()
    prof = None if bk is None else (bk / nb_seen).tolist()
    cls = None
    if cs is not None:
        from sca2 import tokclass as tc
        cls = {tc.NAMES[i]: round((cs[i] / cn[i].clamp(min=1)).item(), 5)
               for i in range(len(tc.NAMES)) if cn[i] > 0}
    return sum(ls) / len(ls), prof, cls


def run(name, m, tr, va, a, device, log):
    m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr)
    n_train = len(tr)

    def lr_at(step, seen):  # warmup in steps; cosine on the fraction of the corpus seen
        w = a.lr * min(1.0, step / max(a.warmup, 1)) if a.warmup else a.lr
        if a.cosine:
            w *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, seen / n_train)))
        return w

    npar = sum(p.numel() for p in m.parameters())
    core = m.core_params() if hasattr(m, "core_params") else 0
    print(f"{name}: {npar} params ({core} in layers)", flush=True)
    it = batches(tr, a.batch, a.block, device)

    for _ in range(a.warm):                       # untimed: absorbs compilation
        x, y = next(it)
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            lg = m(x)
        F.cross_entropy(lg.float().flatten(0, 1), y.flatten()).backward()
        opt.step()
    torch.cuda.synchronize()

    spent, step, seen, nxt = 0.0, 0, 0, 0.0
    while spent < a.seconds:
        try:
            x, y = next(it)
        except StopIteration:
            print(f"{name}: corpus exhausted at step {step}", flush=True); break
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step + 1, seen)
        opt.zero_grad(set_to_none=True)
        with amp_ctx():
            lg = m(x)
        loss = F.cross_entropy(lg.float().flatten(0, 1), y.flatten())
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip if a.clip else float("inf"))
        # Guard the GRADIENT, not the loss. The failure mode this is here for -- a chunked
        # decay kernel that exponentiates the full BT x BT matrix and masks the upper half
        # afterwards -- leaves the FORWARD clean (the masked half is selected away) and
        # blows up only in the backward, where inf x 0 = NaN. The loss stays finite while
        # the gradients are already NaN, so checking loss.item() would see nothing and the
        # arm would burn its remaining hours writing meaningless evals.
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            rec = {"model": name, "seed": a.seed, "step": step + 1, "train_s": round(spent, 1),
                   "tokens": seen, "nonfinite": {"loss": loss.item(), "grad_norm": gn.item()}}
            print(f"!!!! {name}: NON-FINITE at step {step+1} "
                  f"(loss {loss.item()}, grad norm {gn.item()}) -- ABORTING THIS ARM",
                  flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            return m
        opt.step()
        torch.cuda.synchronize()
        spent += time.perf_counter() - t0
        step += 1; seen += a.batch * a.block
        if spent >= nxt or spent >= a.seconds:
            vl, prof, cls = evaluate(m, va, a.batch, a.block, device,
                                     nb=a.eval_batches, buckets=a.pos_buckets,
                                     cls_tab=CLS_TAB)
            rec = {"model": name, "seed": a.seed, "step": step, "train_s": round(spent, 1),
                   "tokens": seen, "train": round(loss.item(), 5), "val": round(vl, 5),
                   "tok_s": round(seen / spent)}
            if prof:
                rec["pos"] = [round(v, 5) for v in prof]
            if cls:
                rec["cls"] = cls
            print(f"  {name} t={rec['train_s']:6.0f}s step {step:6d} "
                  f"{seen/1e6:6.1f}M tok  train {rec['train']:.4f}  val {rec['val']:.4f}  "
                  f"{rec['tok_s']} tok/s"
                  + (f"  pos {prof[0]:.3f}->{prof[-1]:.3f}" if prof else "")
                  + (f"  new {cls['word_new']:.3f} rep {cls['word_rep']:.3f}" if cls else ""),
                  flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            nxt = spent + a.eval_every
    return m


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--eval-every", type=float, default=30, dest="eval_every")
    # 25 batches gives sd(val) = 0.039 nats, the size of every effect chased so
    # far; raise this whenever an architecture comparison is the point.
    p.add_argument("--eval-batches", type=int, default=25, dest="eval_batches")
    # loss by position in the window: the long-context instrument, see evaluate()
    p.add_argument("--pos-buckets", type=int, default=0, dest="pos_buckets")
    p.add_argument("--warm", type=int, default=10)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--block", type=int, default=256)
    p.add_argument("--d", type=int, default=128); p.add_argument("--layers", type=int, default=2)
    p.add_argument("--Mc", type=int, default=64); p.add_argument("--Md", type=int, default=8)
    p.add_argument("--G", type=int, default=8); p.add_argument("--ff", type=int, default=256)
    # value width of BOTH heads' state; None -> d//2. Md costs 9280 params per
    # unit at dv=64, so halving dv is the only way to fund a large Md without
    # also crushing ff (see LayerCfg.dv).
    p.add_argument("--dv", type=int, default=None)
    # Scale of the CONTENT-dependent part of the C head phase, pw = K(h)*theta +
    # p*omega. At 0 the head is a pure lag kernel (ref.freq_grid docstring) and
    # dL/dK is identically 0, so the content path never starts learning. This was
    # hardcoded to 0.0 in every run this file ever produced.
    p.add_argument("--theta-scale", type=float, default=0.0, dest="theta_scale")
    p.add_argument("--trf-ff", type=int, default=367, dest="trf_ff")
    p.add_argument("--variant", default="v3polar_cc"); p.add_argument("--freq", default="rope")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", default="fineweb_bpe16k.pt")
    p.add_argument("--log", default="runs/pretrain.jsonl")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--save", default=None)
    p.add_argument("--label", default=None, help="name this arm in the log")
    # init seed. The batch ORDER is deterministic either way (batches() is
    # sequential), so repeats vary only in initialisation -- which is what the
    # eval-noise question needs, and it keeps the arms seeing identical data.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None)
    p.add_argument("--no-compile", action="store_true", dest="no_compile")
    # gated C head knobs (sca2/arch_gatedc.py), used by --variant gc / gc_cc
    p.add_argument("--c-heads", type=int, default=1, dest="c_heads")
    p.add_argument("--c-decay", action="store_true", dest="c_decay")
    p.add_argument("--c-decay-init", default="gdn", dest="c_decay_init")
    p.add_argument("--c-sepq", action="store_true", dest="c_sepq")
    p.add_argument("--conv", type=int, default=0)
    p.add_argument("--gated-read", action="store_true", dest="gated_read")
    # loss by TOKEN CLASS at every eval (sca2/tokclass.py); --bpe names the
    # tokenizer files the corpus was built with
    # GDN head shape, for --variant gdn* (whole mixer) and hyb* (D-head slot)
    p.add_argument("--Ls", type=int, default=16, help="window of the short dft C head (--variant cshort*)")
    p.add_argument("--rope-base", type=float, default=10000.0, dest="rope_base", help="long-head rope grid base (unaliased range 2*base)")
    p.add_argument("--kv-dk", type=int, default=0, dest="kv_dk",
                   help="key-verification width on the lapa path (0 = off, 16 matches sca2's "
                        "cshort_damphkv). Stores the write key beside the value and gates the "
                        "read on whether it comes back matching the query's key.")
    p.add_argument("--layer-scale", action="store_true", dest="layer_scale",
                   help="learned gain on each residual branch, init 1 (LayerScale). The "
                        "residual stream grows 29x across 10 layers while each branch emits "
                        "a constant norm, so deep layers' relative contribution collapses.")
    p.add_argument("--decay-input", action="store_true", dest="decay_input",
                   help="data-dependent forgetting: lam = lam_max*sigmoid(a_m + Wd(z)_m), a "
                        "function of the token, instead of a constant per mode. GDN's decay "
                        "works this way. Costs d*M parameters.")
    p.add_argument("--beta-init", type=float, default=-2.0, dest="beta_init",
                   help="erase-gate bias at init; sigmoid of it. -2.0 = 0.12 (historical), "
                        "0.0 = 0.5 which is where GDN's starts (its b has no bias).")
    p.add_argument("--conv-silu", action="store_true", dest="conv_silu",
                   help="SiLU after the causal conv, as GDN does on its q/k/v convs. Ours "
                        "was a purely linear convolution.")
    p.add_argument("--long-groups", type=int, default=1, dest="long_groups",
                   help="spectral read weights of the LONG head, per group of value "
                        "channels. This is wg2's experiment, which lost at d=128.")
    p.add_argument("--short-groups", type=int, default=1, dest="short_groups",
                   help="spectral read weights of the SHORT head, per group of value channels. "
                        "1 = one L-tap filter shared by all dv channels (historical); G gives "
                        "the pillar G temporal profiles instead of one. Identical at init.")
    p.add_argument("--beta-groups", type=int, default=1, dest="beta_groups",
                   help="erase-gate granularity: 1 = one scalar per token for all M modes "
                        "(historical), 3 = one per spectral band (slow / persistent-fast / "
                        "damped). Identical at init; 2*d extra parameters.")
    p.add_argument("--kv-gate-pc", action="store_true", dest="kv_gate_pc",
                   help="per-channel key-verification gate: ga/gb become 2*dv vectors instead "
                        "of scalars. Identical at init; 1024 extra parameters at dv=256.")
    p.add_argument("--persist", type=float, default=0.5,
                   help="fraction of long-head modes starting persistent (lambda = 0)")
    p.add_argument("--learn-persist", action="store_true", dest="learn_persist",
                   help="lambda = lam_max*sigmoid(a) instead of softplus(a).clamp(max=lam_max) "
                        "with a hard pin: nothing is pinned, --persist only sets where the split "
                        "STARTS, and there is no zero-gradient region above the cap.")
    p.add_argument("--rope-min-period", type=float, default=None, dest="rope_min_period",
                   help="shortest period in the fast rope grid (None = 2, historical). Set to "
                        "2*Ls to stop spending long-head modes on lags the short head already "
                        "taps exactly -- at M=256/Ls=64 that overlap is 45%% of all modes.")
    p.add_argument("--slow-frac", type=float, default=0.0, dest="slow_frac", help="fraction of long-head modes kept as slow integrators (periods 2..20 x block)")
    p.add_argument("--lam-free", action="store_true", dest="lam_free",
                   help="FREE MODES: lambda = exp(a), nothing pinned at 0 and no cap but the "
                        "fp32 safety ceiling (--lam-ceil, default 55/chunk = 0.43, a memory "
                        "floor of 2.3 tokens). Replaces softplus(a).clamp(max=lam_max) with a "
                        "hard pin, whose realised spectrum on the trained d=1024 checkpoint is "
                        "TWO POINTS: 41-100% of each layer's free modes sit exactly at the "
                        "clamp, where the gradient is zero and no mode ever escapes, and the "
                        "rest are pinned at 0. Use with a WIDE --damp-mem (GDN's measured span "
                        "is 2.5 .. 5.8e6 tokens). --persist is ignored.")
    p.add_argument("--lam-ceil", type=float, default=None, dest="lam_ceil",
                   help="fp32 safety ceiling for --lam-free; default 55/chunk")
    p.add_argument("--lam-max", type=float, default=None, dest="lam_max", help="decay cap of the damped modes; default 1/Ls (window-aligned). Runs before 2026-09-11: 0.125")
    p.add_argument("--damp-mem", default=None, dest="damp_mem", help="init memories lo,hi of the damped modes; default Ls,32*Ls. Runs before 2026-09-11: 64,4096")
    p.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps (0 = none, the campaign's setting)")
    p.add_argument("--cosine", action="store_true", help="cosine lr decay to 10%% of the peak over ONE PASS of the corpus (progress = tokens seen / train tokens)")
    p.add_argument("--gdn-heads", type=int, default=3, dest="gdn_heads")
    p.add_argument("--gdn-head-k", type=int, default=60, dest="gdn_head_k")
    p.add_argument("--gdn-expand-v", type=float, default=1.0, dest="gdn_expand_v")
    p.add_argument("--class-eval", action="store_true", dest="class_eval")
    p.add_argument("--bpe", default="pycode_bpe16k")
    p.add_argument("--clip", type=float, default=0.0,
                   help="global grad-norm clip (0 = off, the d=128 campaign's setting). "
                        "At d=1024 the observed norm at init is 0.6-0.8, so --clip 1.0 is "
                        "inactive in normal operation and only catches a spike -- cheap "
                        "insurance for a long unattended run, applied to every arm alike.")
    p.add_argument("--amp", default="fp32", choices=("fp32", "bf16"),
                   help="autocast dtype of the forward pass, applied to EVERY arm. "
                        "fp32 (default) is the d=128 campaign's setting; bf16 is what "
                        "the Spark runs use (SPARK.md §9). Loss always on fp32 logits.")
    a = p.parse_args(argv)
    global CLS_TAB, AMP
    AMP = torch.bfloat16 if a.amp == "bf16" else None
    if a.class_eval:
        from sca2 import tokclass as tc
        CLS_TAB = tc.load_table(a.bpe)
    device = "cuda"

    z = torch.load(a.data)
    tr, va, V = z["train"], z["val"], z["vocab"]
    print(f"corpus {len(tr)/1e6:.1f}M train / {len(va)/1e6:.1f}M val tokens, vocab {V}")
    print(f"budget {a.seconds:.0f}s per arm, {a.layers} layers, B={a.batch} T={a.block}\n")

    cfg = LayerCfg(a.d, a.Mc, a.Md, a.G, a.ff, freq=a.freq,
                   theta_scale=a.theta_scale, max_len=a.block,
                   dv=a.dv,
                   c_heads=a.c_heads, c_decay=a.c_decay, c_decay_init=a.c_decay_init,
                   c_sepq=a.c_sepq, conv=a.conv,
                   gated_read=a.gated_read,
                   Ls=a.Ls, rope_base=a.rope_base, slow_frac=a.slow_frac,
                   rope_min_period=a.rope_min_period,
                   persist=a.persist, learn_persist=a.learn_persist, kv_dk=a.kv_dk,
                   kv_gate_pc=a.kv_gate_pc, beta_groups=a.beta_groups,
                   short_groups=a.short_groups, long_groups=a.long_groups,
                   conv_silu=a.conv_silu, beta_init=a.beta_init, decay_input=a.decay_input,
                   layer_scale=a.layer_scale,
                   lam_max=a.lam_max, lam_free=a.lam_free, lam_ceil=a.lam_ceil, damp_mem=tuple(float(v) for v in a.damp_mem.split(",")) if a.damp_mem else None,
                   gdn_heads=a.gdn_heads, gdn_head_k=a.gdn_head_k,
                   gdn_expand_v=a.gdn_expand_v)
    log = open(a.log, "a")
    models = {}
    if a.only != "transformer":
        torch.manual_seed(a.seed)
        nm = a.label or "SCA2"
        models[nm] = run(nm, SCA2(V, cfg, a.variant, device, a.layers),
                         tr, va, a, device, log)
    if a.only != "sca2":
        torch.manual_seed(a.seed)
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
