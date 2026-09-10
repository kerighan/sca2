"""Laplace Attention (LapA): a phase-coded, damped-Fourier sequence-mixing layer.

    from lapa import LaplaceAttention, LaplaceConfig, LaplaceLM

    layer = LaplaceAttention(LaplaceConfig(d=512))          # drop-in block: (B,T,d) -> (B,T,d)
    y = layer(x)                                            # parallel prefill
    y, state = layer.prefill(x); y_t, state = layer.step(x_t, state)   # O(1) decode

    lm = LaplaceLM(vocab=32000, d=512, n_layers=12)         # embedding + N layers + LM head
    logits = lm(tokens)                                     # (B,T,V)
    out = lm.generate(prompt, n=100)                        # token-by-token, constant memory

Math, precision policy and provenance: lapa/layer.py docstring and CATCHUP.md.
"""

from .layer import LaplaceAttention, LaplaceConfig, LongHead, ShortHead
from .model import LaplaceLM, LM

__all__ = [
    "LaplaceAttention",
    "LaplaceConfig",
    "LongHead",
    "ShortHead",
    "LaplaceLM",
    "LM",
]
__version__ = "0.1.0"
