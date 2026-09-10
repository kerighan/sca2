"""
torch.compile wrappers.

The eager profile says the layer is launch-latency bound, not bandwidth bound:
78% of D-head forward time is elementwise kernels averaging ~20 us each on
tensors of 0.5-1 MB, which is ~5x what their traffic costs. Decode is worse --
~40 launches to move a few hundred KB. Fusion, not fewer FLOPs, is the lever.

Two compilations, because the two modes want opposite things:
  * prefill  -> default/max-autotune, no cudagraphs: shapes are static per (B,T)
    but the caller may legitimately change them, and autograd must work.
  * step     -> plain fusion only. "reduce-overhead" (cudagraph trees) cannot
    be used here: the state returned by step N is fed back as input to step
    N+1, but cudagraph trees owns those output buffers and overwrites them on
    the next replay ("accessing tensor output of CUDAGraphs that has been
    overwritten"). Graph capture for decode is done explicitly instead, with
    static state buffers -- see sca2/decode.py.
"""
import torch
import torch.nn as nn

# One specialization per (batch, length) exhausts Dynamo's recompile budget fast,
# and once it is exhausted torch.compile falls back to EAGER SILENTLY. That is
# what produced the "batch cliff": at B>=32 the supposedly-compiled numbers were
# eager numbers to three digits (24.45 vs 24.28, 35.74 vs 35.21 us/token).
torch._dynamo.config.cache_size_limit = max(
    64, getattr(torch._dynamo.config, "cache_size_limit", 8))


class CompiledLayer(nn.Module):
    """Wraps an SCA2Layer, preserving the prefill/step contract exactly."""

    def __init__(self, layer, prefill_mode="default", step_mode="default",
                 dynamic=False):
        """prefill_mode="reduce-overhead" turns cudagraphs ON for prefill.

        MEASURED: this does not work as a drop-in. The guess was that the
        objection above is specific to `step` -- that only the RETURNED STATE fed
        back as the next call's input collides with cudagraph trees' buffer
        ownership, and that a training prefill (one call per batch, state
        discarded, static shapes) would be safe. It is not:

            RuntimeError: accessing tensor output of CUDAGraphs that has been
            overwritten by a subsequent run.

        The prefill's outputs are graph-owned too, and they outlive the call --
        they are held for the backward, and pretrain.py interleaves eval forwards
        with training steps. The documented remedy, torch.compiler.
        cudagraph_mark_step_begin() before every invocation, is CALLER-side: it
        changes this layer's contract rather than being free.

        Kept registered (`v3polar_cg`, `v3polarflat_cg`) so the experiment is
        reproducible, but the launch-overhead problem it was meant to solve is
        better addressed inside the layer -- see sca2/fast_dhead.py, which
        removes the chunk loop outright for +8.8% and no contract change.
        """
        super().__init__()
        self.layer = layer
        self.cfg = layer.cfg
        # dynamic=True: one specialization for every shape. Slightly slower per
        # step than a static specialization, but it never falls back, which
        # matters more when sweeping shapes.
        self._prefill = torch.compile(layer.prefill, dynamic=dynamic, mode=prefill_mode)
        # decode must stay static: the graph is captured over fixed-size state
        self._step = torch.compile(layer.step, dynamic=False, mode=step_mode)

    def init_state(self, B, device, dtype=torch.float32):
        st = self.layer.init_state(B, device, dtype)
        # freeze the empty-state branch before any graph is captured -- only the
        # SCA2 heads carry it; other layers (e.g. the Gated DeltaNet baseline)
        # have a different state shape entirely.
        if isinstance(st.get("c"), dict):
            st["c"]["empty"] = False
        return st

    def forward(self, x):
        return self._prefill(x)[0]

    def prefill(self, x, state=None):
        if state is None:
            state = self.init_state(x.size(0), x.device, x.dtype)
        return self._prefill(x, state)

    def step(self, x_t, state):
        return self._step(x_t, state)


def wrap(layer, **kw):
    return CompiledLayer(layer, **kw)


def wrap_dynamic(layer, **kw):
    return CompiledLayer(layer, dynamic=True, **kw)
