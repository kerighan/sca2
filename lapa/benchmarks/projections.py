"""Paired experiment: concatenate existing long-head projection weights.

    OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
      python -m lapa.benchmarks.projections

Weights and optimizer parameter groups are unchanged. Include weight packing,
output splitting/layout conversions and their gradients in every timed call.
"""

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from ..layer import LaplaceAttention, LaplaceConfig, LongHead
from .triton import measure, runner


class PackedLongHead(LongHead):
    """Benchmark-only alternatives; exact state_dict compatibility."""

    def __init__(self, cfg, pack):
        super().__init__(cfg)
        self.pack = pack

    def _project(self, z):
        if self.pack == "separate":
            return super()._project(z)
        if self.pack == "kv":
            kv = F.linear(z, torch.cat((self.K.weight, self.V.weight)))
            k, v = kv.split((self.M, self.dv), dim=-1)
            return k, v, self.bproj(z), None
        if self.pack == "kvg":
            w = torch.cat((self.K.weight, self.V.weight, self.gp.weight))
            bias = torch.cat((self.gp.bias.new_zeros(self.M + self.dv), self.gp.bias))
            k, v, gate = F.linear(z, w, bias).split((self.M, self.dv, 2 * self.dv), dim=-1)
            return k, v, self.bproj(z), gate
        w = torch.cat((self.K.weight, self.V.weight, self.bproj.weight))
        bias = torch.cat((self.bproj.bias.new_zeros(self.M + self.dv), self.bproj.bias))
        n = w.shape[0]
        if self.pack == "kvb_pad":
            pad = (-n) % 64
            w, bias = F.pad(w, (0, 0, 0, pad)), F.pad(bias, (0, pad))
        out = F.linear(z, w, bias)[..., :n]
        return (*out.split((self.M, self.dv, self.bg), dim=-1), None)


def make_models(cfg, packs):
    template = LaplaceAttention(cfg).cuda()
    # Exercise beta's GEMM as well as its constant initialization in validation.
    with torch.no_grad():
        template.long.bproj.weight.normal_(std=0.01)
    models = {}
    for pack in packs:
        model = copy.deepcopy(template)
        head = PackedLongHead(model.cfg, pack).cuda()
        head.load_state_dict(model.long.state_dict())
        model.long = head
        models[pack] = model
    return models


def validate(models, x, previous, scope):
    expected = None
    errors = {}
    for name, model in models.items():
        module = model.long if scope == "long" else model
        module.zero_grad(set_to_none=True)
        inp = x.detach().clone().requires_grad_()
        fn = torch.compile(module.prefill if scope == "long" else module,
                           fullgraph=True, dynamic=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if scope == "long":
                y, state = fn(inp, previous, None)
                loss = y.float().square().mean() + state["s"].square().mean()
            else:
                y = fn(inp)
                loss = y.float().square().mean()
        loss.backward()
        got = {"output": y.detach(), "input": inp.grad}
        if scope == "long":
            got["state"] = state["s"].detach()
        got.update({n: p.grad for n, p in module.named_parameters()})
        if expected is None:
            expected = {k: v.clone() if v is not None else None for k, v in got.items()}
        grad_scale = max(v.abs().max().item() for k, v in expected.items()
                         if v is not None and k not in ("output", "state"))
        deviations = {}
        for key, ref in expected.items():
            if ref is None:
                assert got[key] is None, key
                continue
            den = ref.abs().max().item() if key in ("output", "state") else grad_scale
            error = (ref - got[key]).abs().max().item() / max(den, 1e-12)
            assert error <= 0.06, (name, key, error)
            deviations[key] = error
        errors[name] = deviations
        module.zero_grad(set_to_none=True)
        print("CHECK", name, max(deviations.items(), key=lambda item: item[1]), flush=True)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", default="separate,kv,kvb,kvb_pad")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--dv", type=int, default=384)
    parser.add_argument("--kv-dk", type=int, default=0)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--ff", type=int, default=5800)
    parser.add_argument("--short-window", type=int, default=128)
    parser.add_argument("--conv-silu", action="store_true")
    parser.add_argument("--beta-write", action="store_true")
    parser.add_argument("--scopes", default="long,layer")
    parser.add_argument("--modes", default="train,step")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()
    packs = args.packs.split(",")
    if packs[0] != "separate" or not set(packs) <= {"separate", "kv", "kvb", "kvb_pad", "kvg"}:
        parser.error("packs must start with separate, followed by kv/kvb/kvb_pad/kvg")
    if "kvg" in packs and args.kv_dk:
        parser.error("kvg requires the GDN gate (--kv-dk 0)")
    if not set(args.scopes.split(",")) <= {"long", "layer"}:
        parser.error("unknown scope")
    if not set(args.modes.split(",")) <= {"train", "step"}:
        parser.error("unknown mode")
    if args.rounds < 2 or args.iters < 1:
        parser.error("at least two rounds and one iteration are required")
    if min(args.tokens, args.dv, args.groups, args.ff, args.short_window) < 1 or args.kv_dk < 0:
        parser.error("dimensions must be positive and kv-dk nonnegative")
    if (args.dv + args.kv_dk) % args.groups:
        parser.error("dv + kv-dk must be divisible by groups")
    torch._dynamo.config.cache_size_limit = 256
    torch.manual_seed(42)
    cfg = LaplaceConfig(d=1024, M=256, dv=args.dv, L=args.short_window, ff=args.ff,
                        chunk=128, rope_base=1000., slow_frac=.25, max_len=args.tokens,
                        kv_dk=args.kv_dk, conv=4, conv_silu=args.conv_silu,
                        beta_write=args.beta_write, gdn_gate=args.kv_dk == 0, lam_free=True,
                        mem_range=(4., 20000.), long_groups=args.groups, layer_scale=True,
                        ls_mix_init=.25, ls_ff_init=.5, w_antipodal=.1, long_path="triton_scan")
    models = make_models(cfg, packs)
    x = torch.randn(8, args.tokens, 1024, device="cuda")
    previous = torch.zeros(8, 1024, device="cuda")
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    print(cfg, flush=True)
    for scope in args.scopes.split(","):
        checks = validate(models, x, previous, scope) if not args.skip_check else None
        for mode in args.modes.split(","):
            if scope == "long" and mode == "step":
                continue
            print(scope, mode, flush=True)
            arms = {name: runner(model, x, previous, scope, mode) for name, model in models.items()}
            result = measure(arms, args.rounds, args.iters)
            print("RESULT", json.dumps(dict(scope=scope, mode=mode, config=vars(args),
                                             checks=checks, arms=result)), flush=True)


if __name__ == "__main__":
    main()
