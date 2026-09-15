"""Experimental long-head Triton inverse/adjoint with CPU/fp64 fallback."""

import torch


def reference_inverse(gram, beta=None):
    lower = gram.tril(-1)
    if beta is not None:
        lower = beta * lower
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return torch.linalg.solve_triangular(
        eye + lower, eye.expand_as(gram), upper=False, unitriangular=True
    )


class _Inverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gram, beta):
        from .triton_solve_kernel import inverse_forward

        out = inverse_forward(gram, beta)
        ctx.shared = beta is not None
        ctx.save_for_backward(gram, out, *(() if beta is None else (beta,)))
        return out

    @staticmethod
    def backward(ctx, grad):
        gram, inverse, *rest = ctx.saved_tensors
        from .triton_solve_kernel import inverse_backward

        return inverse_backward(gram, inverse, grad, rest[0] if ctx.shared else None)


def triangular_inverse(gram, beta=None):
    """Inverse(I + beta * tril(gram, -1)).

    gram: (..., C, C); beta: (..., C, 1), or None for already gated codes.
    CUDA fp32 chunks up to 128 use Triton. Other sizes/devices and fp64 use
    PyTorch. This prototype does not yet establish a training speedup.
    """
    if gram.ndim < 2 or gram.shape[-2] != gram.shape[-1] or gram.shape[-1] == 0:
        raise ValueError("gram must contain nonempty square matrices")
    if gram.dtype not in (torch.float32, torch.float64):
        raise ValueError("the triangular solve requires float32 or float64")
    if beta is not None and (
        beta.shape != gram.shape[:-1] + (1,)
        or beta.dtype != gram.dtype
        or beta.device != gram.device
    ):
        raise ValueError("beta must match gram's batch, row, dtype and device")
    if not gram.is_cuda or gram.dtype != torch.float32 or gram.shape[-1] > 128:
        return reference_inverse(gram, beta)
    return _Inverse.apply(
        gram.contiguous(), None if beta is None else beta.contiguous()
    )
