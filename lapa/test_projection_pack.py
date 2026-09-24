"""The experimental packed projections preserve the fp64 layer and adjoint."""

import unittest

import torch

from .benchmarks.projections import PackedLongHead
from .layer import LaplaceConfig, LongHead


class TestProjectionPack(unittest.TestCase):
    def test_outputs_state_and_gradients(self):
        torch.manual_seed(73)
        cfg = LaplaceConfig(d=16, M=9, dv=7, L=4, chunk=4, gdn_gate=True)
        ref = LongHead(cfg).double()
        with torch.no_grad():
            ref.bproj.weight.normal_(std=0.1)
        x = torch.randn(2, 11, 16, dtype=torch.float64)
        previous = torch.randn(2, 16, dtype=torch.float64)
        carry = ref.init_state(2, "cpu")
        carry["s"].normal_(std=0.1)
        carry["pos"].fill_(3)
        expected = None
        for pack in ("separate", "kv", "kvb", "kvb_pad", "kvg"):
            with self.subTest(pack=pack):
                model = PackedLongHead(cfg, pack).double() if pack != "separate" else ref
                model.load_state_dict(ref.state_dict(), strict=True)
                self.assertEqual(dict(model.named_parameters()).keys(), dict(ref.named_parameters()).keys())
                z, prev = x.clone().requires_grad_(), previous.clone().requires_grad_()
                st = {k: v.clone() for k, v in carry.items()}
                st["s"].requires_grad_()
                y, final = model.prefill(z, prev, st)
                (y.square().mean() + final["s"].square().mean()).backward()
                got = (y, final["s"], z.grad, prev.grad, st["s"].grad,
                       *(p.grad for p in model.parameters()))
                if expected is None:
                    expected = got
                for a, b in zip(got, expected):
                    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
