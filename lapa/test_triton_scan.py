"""End-to-end checks of the fused long-head scan against the batched reference."""

import copy
import math
import unittest

import torch

from .layer import LaplaceConfig, LongHead


def _reference(kk, qk, fq, k2, w, v, beta, s, d1, dc, gt):
    """The chunk loop written out, in the reference's dtype discipline."""
    b, k, c, r = kk.shape
    d, g = v.shape[-1], fq.shape[2]
    dt = kk.dtype
    sn = s
    outs = []
    for n in range(k):
        s0 = (sn * d1).to(dt)
        read = (qk[:, n] @ s0).float() / (r // 2)
        e = (w[:, n] @ (v[:, n] - beta[:, n] * read)).to(dt)
        eg = e.view(b, c, g, d // g).transpose(1, 2)
        sg = s0.view(b, r, g, d // g).transpose(1, 2)
        o = (k2[:, n] @ eg + fq[:, n] @ sg).float().transpose(1, 2)
        outs.append(o.reshape(b, 2 * c, d))
        sn = sn * dc + (kk[:, n].transpose(-1, -2) @ e).float() * gt
    o = torch.stack(outs, 1)
    return torch.cat([o[:, :, :c], o[:, :, c:]], -1).reshape(b, k * c, 2 * d) / (r // 2), sn


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Each shape is its own graph; the default limit of 8 is reached when
        # this module runs after the other suites in one process.
        torch._dynamo.config.cache_size_limit = 256
    def check(self, c, modes, dvi, groups, amp, dtype=torch.float32,
              compile_=False, path="triton_scan", near_ceiling=False, chunks=2,
              **features):
        torch.manual_seed(42)
        cfg = LaplaceConfig(d=32, M=modes, dv=dvi - 4, kv_dk=4, L=4, chunk=c,
                            long_groups=groups, lam_free=True,
                            mem_range=(4.0, 20000.0), **features)
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
            fn = (torch.compile(model.prefill, fullgraph=True, dynamic=False)
                  if compile_ else model.prefill)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                y, s = fn(z, zp, st)
                loss = y.square().mean() + s["s"].square().mean()
            loss.backward()
            results.append(dict(output=y, state=s["s"], x=z.grad, prev=zp.grad,
                                carry=st["s"].grad,
                                **{n: p.grad for n, p in model.named_parameters()}))
        self.assertFalse(torch.equal(carry["s"], results[1]["state"]),
                         "the incoming state must not be mutated")
        tol = 1e-10 if dtype == torch.float64 else (0.06 if amp else 3e-4)
        scale = max(v.abs().max().item() for k, v in results[0].items()
                    if k not in ("output", "state") and v is not None)
        errors = {}
        for name, expected in results[0].items():
            got = results[1][name]
            if expected is None:
                self.assertIsNone(got)
                continue
            den = expected.abs().max().item() if name in ("output", "state") else scale
            err = (got - expected).abs().max().item() / max(den, 1e-12)
            errors[name] = err
            self.assertLessEqual(err, tol, f"{name}: {err:.3e}, C={c}, M={modes}, "
                                           f"D={dvi}, NG={groups}, amp={amp}")
        print("PASS", c, modes, dvi, groups, amp, dtype, "compiled", compile_,
              "worst", max(errors.items(), key=lambda kv: kv[1]), flush=True)

    def test_fp32(self):
        for c, m, d, g in ((8, 16, 16, 1), (7, 19, 18, 2), (32, 32, 36, 2),
                           (128, 256, 272, 2)):
            self.check(c, m, d, g, False)

    def test_group_counts(self):
        """Read groups beyond the trained NG=2: the grouped column blocking and
        the static group reduction in the dKk adjoint both depend on it."""
        for g in (4, 8):
            self.check(8, 16, 16, g, False)
            self.check(128, 256, 272, g, False)
            self.check(128, 256, 272, g, True, compile_=True, chunks=8)
        self.check(7, 19, 20, 4, False)
        self.check(1, 19, 20, 4, False)
        self.check(8, 16, 16, 4, False, torch.float64)
        self.check(8, 16, 16, 4, False, beta_groups=3, decay_input=True)

    def test_bf16(self):
        for c, m, d, g in ((8, 16, 16, 1), (32, 32, 36, 2), (128, 256, 272, 2)):
            self.check(c, m, d, g, True)

    def test_fp64_fallback(self):
        self.check(8, 16, 16, 2, False, torch.float64)

    def test_compile(self):
        self.check(8, 16, 16, 2, True, compile_=True)
        self.check(8, 16, 16, 2, False, compile_=True)

    def test_fallback_features(self):
        self.check(8, 16, 16, 2, False, beta_groups=3, decay_input=True)

    def test_decay_ceiling(self):
        self.check(1, 19, 18, 2, False, near_ceiling=True)
        self.check(7, 19, 18, 2, False, near_ceiling=True)

    def test_training_shape(self):
        self.check(128, 256, 272, 2, True, compile_=True, chunks=8)

    def test_each_input_gradient(self):
        """Every input of the fused node against the written-out chunk loop."""
        from .triton_scan import long_chunk
        from .triton_product import code_product
        from .triton_solve import triangular_inverse

        torch.manual_seed(123)
        names = ("kk", "qk", "fq", "v", "beta", "s", "d1", "dc", "gt")
        for (b, k, c, r, d, g), dtype in [(s, t) for s in ((2, 3, 7, 18, 10, 2),
                                                           (2, 3, 7, 18, 12, 4))
                                          for t in (torch.float32, torch.bfloat16)]:
            rand = lambda shape, dt: (
                torch.randn(shape, device="cuda", dtype=dt) * 0.1
            ).requires_grad_()
            args = (rand((b, k, c, r), dtype), rand((b, k, c, r), dtype),
                    rand((b, k, g, c, r), dtype), rand((b, k, c, d), torch.float32),
                    (torch.rand((b, k, c, 1), device="cuda") * 0.3).requires_grad_(),
                    rand((b, r, d), torch.float32),
                    *[(torch.rand((r, 1), device="cuda") * 0.4 + 0.4).requires_grad_()
                      for _ in range(3)])
            kk, qk, fq, v, beta, s, d1, dc, gt = args
            # The reference rebuilds the full read-code block from the compact one.
            c1, c2 = fq[..., : r // 2], fq[..., r // 2:]
            full = torch.cat([fq, torch.cat([-c2, c1], -1)], -2)
            gram = code_product(qk.unsqueeze(2), kk, gram=True).squeeze(2)
            w = triangular_inverse(gram, beta)
            k2 = code_product(full, kk)
            ref, rs = _reference(kk, qk, full, k2, w, v, beta, s, d1, dc, gt)
            got, gs = long_chunk(*args)
            do, ds = torch.randn_like(ref), torch.randn_like(rs)
            rg = torch.autograd.grad((ref, rs), args, (do, ds), retain_graph=True)
            tg = torch.autograd.grad((got, gs), args, (do, ds))
            tol = 3e-4 if dtype == torch.float32 else 0.03
            for name, a, z in zip(names, tg, rg):
                err = ((a.float() - z.float()).abs().max()
                       / z.float().abs().max().clamp_min(1e-12)).item()
                self.assertLessEqual(err, tol, f"{name} {dtype}: {err:.3e}")
            for name, a, z in (("output", got, ref), ("state", gs, rs)):
                err = ((a.float() - z.float()).abs().max()
                       / z.float().abs().max()).item()
                self.assertLessEqual(err, tol, f"{name} {dtype}: {err:.3e}")
            print("PASS gradients", dtype, "NG", g, flush=True)


if __name__ == "__main__":
    unittest.main()
