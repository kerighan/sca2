"""End-to-end checks of the fused state loop against the batched reference."""

import copy
import math
import unittest
import torch
from .layer import LaplaceConfig, LongHead


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStateLoop(unittest.TestCase):
    def check(
        self,
        c,
        modes,
        dvi,
        groups,
        amp,
        dtype=torch.float32,
        compile_=False,
        path="triton_fused",
        near_ceiling=False,
        chunks=2,
        **features,
    ):
        torch.manual_seed(42)
        cfg = LaplaceConfig(
            d=32,
            M=modes,
            dv=dvi - 4,
            kv_dk=4,
            L=4,
            chunk=c,
            long_groups=groups,
            lam_free=True,
            mem_range=(4.0, 20000.0),
            **features,
        )
        ref = LongHead(cfg).cuda().to(dtype)
        if near_ceiling:
            with torch.no_grad():
                ref.lam_raw.fill_(math.log(ref.lam_ceil * 0.999))
        opt = copy.deepcopy(ref)
        opt.cfg.long_path = path
        x = torch.randn(2, chunks * c + 3, 32, device="cuda", dtype=dtype)
        prev = torch.randn(2, 32, device="cuda", dtype=dtype)
        carry = ref.init_state(2, "cuda")
        carry["s"].normal_(std=0.2)
        carry["pos"].fill_(21)
        results = []
        for model in (ref, opt):
            z = x.clone().requires_grad_()
            zp = prev.clone().requires_grad_()
            st = {k: v.clone() for k, v in carry.items()}
            st["s"].requires_grad_()
            fn = (
                torch.compile(model.prefill, fullgraph=True, dynamic=False)
                if compile_
                else model.prefill
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y, s = fn(z, zp, st)
                loss = y.square().mean() + s["s"].square().mean()
            loss.backward()
            results.append(
                dict(
                    output=y,
                    state=s["s"],
                    x=z.grad,
                    prev=zp.grad,
                    carry=st["s"].grad,
                    **{n: p.grad for n, p in model.named_parameters()},
                )
            )
        tol = 1e-10 if dtype == torch.float64 else (0.06 if amp else 3e-4)
        scale = max(
            v.abs().max().item()
            for k, v in results[0].items()
            if k not in ("output", "state") and v is not None
        )
        errors = {}
        for name, expected in results[0].items():
            got = results[1][name]
            if expected is None:
                self.assertIsNone(got)
                continue
            den = expected.abs().max().item() if name in ("output", "state") else scale
            err = (got - expected).abs().max().item() / max(den, 1e-12)
            errors[name] = err
            self.assertLessEqual(
                err,
                tol,
                f"{name}: {err:.3e}, C={c}, M={modes}, D={dvi}, NG={groups}, amp={amp}",
            )
        print(
            "PASS",
            c,
            modes,
            dvi,
            groups,
            amp,
            dtype,
            "compiled",
            compile_,
            "worst",
            max(errors.items(), key=lambda kv: kv[1]),
            flush=True,
        )

    def test_fp32(self):
        for c, m, d, g in (
            (8, 16, 16, 1),
            (7, 19, 18, 2),
            (32, 32, 36, 2),
            (128, 256, 272, 2),
        ):
            self.check(c, m, d, g, False)

    def test_bf16(self):
        for c, m, d, g in ((8, 16, 16, 1), (32, 32, 36, 2), (128, 256, 272, 2)):
            self.check(c, m, d, g, True)

    def test_fp64_fallback(self):
        self.check(8, 16, 16, 2, False, torch.float64)

    def test_compile(self):
        self.check(8, 16, 16, 2, True, compile_=True)

    def test_codes_path(self):
        self.check(32, 32, 36, 2, False, path="triton_codes")
        self.check(32, 32, 36, 2, True, path="triton_codes", compile_=True)

    def test_fallback_features(self):
        self.check(8, 16, 16, 2, False, beta_groups=3, decay_input=True)

    def test_decay_ceiling(self):
        self.check(1, 19, 18, 2, False, near_ceiling=True)
        self.check(7, 19, 18, 2, False, near_ceiling=True)

    def test_training_shape(self):
        self.check(128, 256, 272, 2, True, compile_=True, chunks=8)

    def test_each_state_input_gradient(self):
        from .triton_state import state_loop

        torch.manual_seed(123)
        b, k, c, r, d, g = 2, 3, 7, 18, 10, 2
        for dtype in (torch.float32, torch.bfloat16):

            def rand(shape, dt):
                return (
                    torch.randn(shape, device="cuda", dtype=dt) * 0.1
                ).requires_grad_()

            args = (
                rand((b, k, c, r), dtype),
                rand((b, k, c, r), dtype),
                rand((b, k, g, 2 * c, r), dtype),
                rand((b, k, g, 2 * c, c), dtype),
                rand((b, k, c, c), torch.float32),
                rand((b, k, c, d), torch.float32),
                rand((b, k, c, 1), torch.float32),
                rand((b, r, d), torch.float32),
                *[rand((r, 1), torch.float32) for _ in range(3)],
            )
            kk, qk, fq, k2, w, v, beta, s, d1, dc, gt = args
            sn = s
            outs = []
            for n in range(k):
                s0 = (sn * d1).to(dtype)
                read = (qk[:, n] @ s0).float() / (r // 2)
                e = (w[:, n] @ (v[:, n] - beta[:, n] * read)).to(dtype)
                eg = e.view(b, c, g, d // g).transpose(1, 2)
                sg = s0.view(b, r, g, d // g).transpose(1, 2)
                o = (
                    (k2[:, n] @ eg + fq[:, n] @ sg)
                    .float()
                    .transpose(1, 2)
                    .reshape(b, 2 * c, d)
                )
                outs.append(o)
                sn = sn * dc + (kk[:, n].transpose(-1, -2) @ e).float() * gt
            ref = torch.stack(outs, 1)
            got, gs = state_loop(*args)
            do = torch.randn_like(ref)
            ds = torch.randn_like(sn)
            rg = torch.autograd.grad((ref, sn), args, (do, ds))
            tg = torch.autograd.grad((got, gs), args, (do, ds))
            tol = 3e-4 if dtype == torch.float32 else 0.03
            for name, a, z in zip(
                ("kk", "qk", "fq", "k2", "w", "v", "beta", "s", "d1", "dc", "gt"),
                tg,
                rg,
            ):
                error = (
                    a.float() - z.float()
                ).abs().max() / z.float().abs().max().clamp_min(1e-12)
                self.assertLessEqual(error.item(), tol, f"{name} {dtype}: {error}")


if __name__ == "__main__":
    unittest.main()
