"""Run with python -m unittest lapa.test_triton_solve (CUDA tests auto-skip)."""
import copy
import unittest

import torch

from .layer import LaplaceConfig, LongHead
from .triton_solve import reference_inverse, triangular_inverse


class TestInverse(unittest.TestCase):
    def check_inverse(self, device, dtype):
        torch.manual_seed(42)
        tol = 1e-10 if dtype == torch.float64 else 3e-4
        for c in (1, 7, 32, 128, 129):
            for shared in (False, True):
                with self.subTest(device=device, dtype=dtype, c=c, shared=shared):
                    # Noncontiguous inputs; upper triangle/diagonal are deliberately nonzero.
                    g = (torch.randn(2, 2, c, c, device=device, dtype=dtype) * .05)
                    g = g.transpose(-1, -2).requires_grad_()
                    b = torch.rand(2, 2, c, 1, device=device, dtype=dtype, requires_grad=True)
                    args = (g, b) if shared else (g,)
                    out, ref = triangular_inverse(*args), reference_inverse(*args)
                    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)
                    grad = torch.randn_like(out)
                    got = torch.autograd.grad(out, args, grad)
                    expected = torch.autograd.grad(ref, args, grad)
                    for a, e in zip(got, expected):
                        torch.testing.assert_close(a, e, atol=tol, rtol=tol)
                    torch.testing.assert_close(got[0].triu(), torch.zeros_like(g))

    def test_cpu(self):
        for dtype in (torch.float64, torch.float32):
            self.check_inverse('cpu', dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda(self):
        for dtype in (torch.float64, torch.float32):
            self.check_inverse('cuda', dtype)

    def check_layer(self, device, dtype, amp=False):
        torch.manual_seed(7)
        for ng, bg, di in ((1, 1, False), (2, 1, False), (2, 3, True)):
            with self.subTest(device=device, dtype=dtype, ng=ng, bg=bg, di=di, amp=amp):
                cfg = LaplaceConfig(d=16, M=16, dv=12, kv_dk=4, L=4, chunk=8,
                                    long_groups=ng, beta_groups=bg, decay_input=di,
                                    lam_free=True, long_path='batched')
                ref = LongHead(cfg).to(device=device, dtype=dtype)
                opt = copy.deepcopy(ref)
                opt.cfg.long_path = 'triton'
                # Two full chunks plus a ragged tail, nonzero carry and offset.
                z = torch.randn(2, 19, 16, device=device, dtype=dtype)
                zp = torch.randn(2, 16, device=device, dtype=dtype)
                state = ref.init_state(2, device)
                state['s'].normal_(std=.1)
                state['pos'].fill_(17)
                results = []
                for model in (ref, opt):
                    x = z.clone().requires_grad_()
                    carry = {k: v.clone() for k, v in state.items()}
                    carry['s'].requires_grad_()
                    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=amp):
                        y, st = model.prefill(x, zp, carry)
                        loss = y.square().mean() + st['s'].square().mean()
                    loss.backward()
                    results.append((y, st['s'], x.grad, carry['s'].grad,
                                    *[p.grad for p in model.parameters()]))
                tol = 1e-10 if dtype == torch.float64 else (6e-2 if amp else 3e-4)
                for got, expected in zip(results[1], results[0]):
                    if expected is None:
                        self.assertIsNone(got)
                    else:
                        torch.testing.assert_close(got, expected, atol=tol, rtol=tol)

    def test_layer_cpu(self):
        for dtype in (torch.float64, torch.float32):
            self.check_layer('cpu', dtype)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_layer_cuda(self):
        for dtype in (torch.float64, torch.float32):
            self.check_layer('cuda', dtype)
        self.check_layer('cuda', torch.float32, amp=True)


if __name__ == '__main__':
    unittest.main()
