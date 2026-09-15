"""Fused long-head state loop. CUDA fp32/bf16, scalar beta, fixed decay.

Forward: four launches per chunk, with state decay and dtype conversions
inside GEMMs. Backward: three sequential launches per chunk, then batched
code/weight/ramp gradients. All math in the custom backward is Triton.
"""

import torch


class _StateLoop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kk, qk, fq, k2, w, v, beta, s, d1, dc, gt, save):
        from .triton_state_kernel import forward

        o, states, h, read, write, e, s0 = forward(
            kk, qk, fq, k2, w, v, beta, s, d1, dc, gt, save
        )
        ctx.save_for_backward(
            kk, qk, fq, k2, w, beta, s, d1, dc, gt, states, h, read, write, e, s0
        )
        return o, states[:, -1]

    @staticmethod
    def backward(ctx, do, ds):
        from .triton_state_kernel import backward

        return (*backward(*ctx.saved_tensors, do.contiguous(), ds.contiguous()), None)


def state_loop(kk, qk, fq, k2, w, v, beta, s, d1, dc, gt):
    """Inputs are batched over (B,K); fq/k2 have an explicit group axis.

    Output is (B,K,2C,V), with real and imaginary rows stacked; the caller
    packs/scales the output. Input state and fp32 ramp factors are not mutated.
    """
    return _StateLoop.apply(
        kk,
        qk,
        fq,
        k2,
        w.contiguous(),
        v.contiguous(),
        beta,
        s.contiguous(),
        d1,
        dc,
        gt,
        torch.is_grad_enabled(),
    )
