"""Write-gate recurrence and chunk/decode gradient equivalence on CPU."""

import copy
import unittest

import torch

from .layer import LaplaceConfig, LongHead


class TestBetaWrite(unittest.TestCase):
    def test_config_mapping_and_scalar_requirement(self):
        from sca2.arch_lapa import config_from
        from sca2.ref import LayerCfg

        self.assertFalse(LaplaceConfig().beta_write)
        cfg = config_from(LayerCfg(32, 16, 4, 2, 48, beta_write=True))
        self.assertTrue(cfg.beta_write)
        with self.assertRaisesRegex(ValueError, "beta_groups=1"):
            LongHead(LaplaceConfig(beta_write=True, beta_groups=3))

    def test_recurrence(self):
        # Independent one-token state update, including a nonzero memory and
        # stored key channels. Check beta's derivative as well as the value.
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                torch.manual_seed(81)
                head = LongHead(LaplaceConfig(d=8, M=6, dv=4, kv_dk=2,
                                             beta_write=enabled)).double()
                with torch.no_grad():
                    head.bproj.weight.normal_(std=0.2)
                z, prev = torch.randn(2, 8, dtype=torch.float64), torch.randn(2, 8, dtype=torch.float64)
                st = head.init_state(2, "cpu")
                st["s"].normal_()
                st["pos"].fill_(7)
                _, got = head.step(z, prev, st)
                phase = head.K(prev) * head.theta + st["pos"] * head.omega
                key = torch.cat((phase.cos(), phase.sin()), -1)
                decayed = st["s"] * head.lam().neg().exp().repeat(2)[None, :, None]
                read = (key[:, :, None] * decayed).sum(1) / head.M
                value = torch.cat((head.V(z), head.Kv(prev)), -1)
                beta = head.bproj(z).sigmoid()
                delta = beta * (value - read) if enabled else value - beta * read
                expected = decayed + key[:, :, None] * delta[:, None, :]
                torch.testing.assert_close(got["s"], expected, atol=1e-12, rtol=1e-12)
                for actual, ref in zip(
                    torch.autograd.grad(got["s"].square().sum(), (head.bproj.weight, head.bproj.bias)),
                    torch.autograd.grad(expected.square().sum(), (head.bproj.weight, head.bproj.bias)),
                ):
                    torch.testing.assert_close(actual, ref, atol=1e-11, rtol=1e-11)

    def test_chunk_decode_gradients(self):
        for decay_input in (False, True):
            for path in ("chunk", "batched"):
                with self.subTest(decay_input=decay_input, path=path):
                    torch.manual_seed(82)
                    cfg = LaplaceConfig(d=8, M=6, dv=6, kv_dk=2, long_groups=2,
                                        chunk=4, beta_write=True, decay_input=decay_input,
                                        long_path=path)
                    head = LongHead(cfg).double()
                    with torch.no_grad():
                        head.bproj.weight.normal_(std=0.2)
                        if decay_input:
                            head.lam_proj.weight.normal_(std=0.2)
                    decoded = copy.deepcopy(head)
                    x = torch.randn(2, 11, 8, dtype=torch.float64)
                    prev = torch.randn(2, 8, dtype=torch.float64)
                    carry = head.init_state(2, "cpu")
                    carry["s"].normal_(std=0.2)
                    carry["pos"].fill_(5)
                    results = []
                    for model, decode in ((head, False), (decoded, True)):
                        z, zp = x.clone().requires_grad_(), prev.clone().requires_grad_()
                        st = {k: v.clone() for k, v in carry.items()}
                        st["s"].requires_grad_()
                        incoming = st["s"]
                        if decode:
                            ys = []
                            for t in range(z.shape[1]):
                                y, st = model.step(z[:, t], zp if t == 0 else z[:, t - 1], st)
                                ys.append(y)
                            y = torch.stack(ys, 1)
                        else:
                            y, st = model.prefill(z, zp, st)
                        (y.square().mean() + st["s"].square().mean()).backward()
                        results.append([y, st["s"], z.grad, zp.grad, incoming.grad,
                                        *[p.grad for p in model.parameters()]])
                    for actual, ref in zip(*results):
                        torch.testing.assert_close(actual, ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
