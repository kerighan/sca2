"""Paired compiled benchmarks at the TRITON.md training shape.

    python -m lapa.benchmarks.triton --paths batched,triton_codes,triton_fused

This measures one attention layer with synthetic inputs, not a full LM.
The optimizer arm uses lr=0 to time AdamW without changing the weights.
"""

import argparse
import copy
import json
import statistics
import time

import torch
from ..layer import LaplaceAttention, LaplaceConfig


def measure(arms, rounds, iters):
    names = list(arms)
    for name, fn in arms.items():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        print(f"  ready: {name}", flush=True)
    times = {name: [] for name in names}
    for r in range(rounds):
        for name in names[:: 1 if r % 2 == 0 else -1]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                arms[name]()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) * 1000 / iters)
    base = names[0]
    return {
        name: {
            "ms": statistics.median(ts),
            "paired_ratio": statistics.median(t / b for t, b in zip(ts, times[base])),
            "rounds_ms": ts,
        }
        for name, ts in times.items()
    }


def runner(model, x, previous, scope, mode):
    module = model.long if scope == "long" else model
    module.train(mode != "forward")
    inp = x.clone().requires_grad_(mode != "forward")
    fn = torch.compile(
        module.prefill if scope == "long" else module, fullgraph=True, dynamic=False
    )
    optimizer = (
        torch.optim.AdamW(module.parameters(), lr=0.0, fused=True)
        if mode == "step"
        else None
    )

    def run():
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            inp.grad = None
        with torch.set_grad_enabled(mode != "forward"), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            out = fn(inp, previous, None)[0] if scope == "long" else fn(inp)
            loss = out.float().square().mean() if mode != "forward" else None
        if loss is not None:
            loss.backward()
        if optimizer is not None:
            optimizer.step()

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default="batched,triton_codes,triton_fused")
    parser.add_argument("--scopes", default="long,layer")
    parser.add_argument("--modes", default="forward,train,step")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    paths = args.paths.split(",")
    if not set(paths) <= {"batched", "triton", "triton_codes", "triton_fused", "triton_scan"}:
        parser.error("unknown path")
    if not set(args.scopes.split(",")) <= {"long", "layer"}:
        parser.error("scopes must be long and/or layer")
    if not set(args.modes.split(",")) <= {"forward", "train", "step"}:
        parser.error("modes must be forward, train and/or step")
    if args.rounds < 2 or args.iters < 1:
        parser.error("at least two rounds and one iteration are required")
    torch._dynamo.config.cache_size_limit = 256
    torch.manual_seed(42)
    cfg = LaplaceConfig(
        d=1024,
        M=256,
        dv=256,
        L=64,
        ff=4096,
        chunk=128,
        rope_base=1000.0,
        slow_frac=0.25,
        max_len=1024,
        kv_dk=16,
        conv=4,
        lam_free=True,
        mem_range=(4.0, 20000.0),
        long_groups=2,
        layer_scale=True,
        ls_mix_init=0.25,
        ls_ff_init=0.5,
        ls_mix_per_channel=False,
        w_antipodal=0.1,
    )
    template = LaplaceAttention(cfg).cuda()
    models = {path: copy.deepcopy(template) for path in paths}
    for path, model in models.items():
        model.cfg.long_path = path
    x = torch.randn(8, 1024, 1024, device="cuda")
    previous = torch.zeros(8, 1024, device="cuda")
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    print(cfg, flush=True)
    for scope in args.scopes.split(","):
        for mode in args.modes.split(","):
            if scope == "long" and mode == "step":
                continue
            print(f"{scope} / {mode}", flush=True)
            arms = {
                path: runner(model, x, previous, scope, mode)
                for path, model in models.items()
            }
            result = measure(arms, args.rounds, args.iters)
            print(
                "RESULT",
                json.dumps({"scope": scope, "mode": mode, "arms": result}),
                flush=True,
            )


if __name__ == "__main__":
    main()
