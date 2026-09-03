"""
Manual CUDA-graph decoder.

Decode is pure launch overhead: ~40 tiny kernels to move a few hundred KB, at
~20 us of launch latency each. torch.compile's own cudagraph mode can't be used
because the state is a loop-carried dependency it would clobber (see
compiled.py), so capture the graph by hand with STATIC state buffers and put the
state write-back inside the graph. One replay per token, and the recurrence
advances in place.
"""
import torch


def _pairs(old, new):
    """Walk two same-shaped nested state dicts together, yielding aligned
    (destination tensor, source tensor) leaves."""
    for k, v in old.items():
        if isinstance(v, dict):
            yield from _pairs(v, new[k])
        elif torch.is_tensor(v):
            yield v, new[k]


def _leaves(state):
    for k, v in state.items():
        if isinstance(v, dict):
            yield from _leaves(v)
        elif torch.is_tensor(v):
            yield v


def _clone_state(state):
    return {k: (_clone_state(v) if isinstance(v, dict) else v) for k, v in state.items()}


def _check_capturable(state, path=""):
    """A host-side *value* in the state is frozen at capture time and silently
    stops advancing on replay -- e.g. a python-int position makes every token
    decode at position 0. Booleans are exempt: they select a branch, which is
    legitimately constant for the whole decode.
    """
    for k, v in state.items():
        p = f"{path}{k}"
        if isinstance(v, dict):
            _check_capturable(v, p + ".")
        elif isinstance(v, bool) or torch.is_tensor(v):
            continue
        else:
            raise TypeError(
                f"state[{p!r}] is a host-side {type(v).__name__} ({v!r}); CUDA graph "
                f"capture would freeze it and it would never advance. Store it as a "
                f"0-dim tensor instead (see CHeadQuad.init_state).")


class GraphDecoder:
    """Wraps layer.step into a single replayable CUDA graph.

    Usage:
        dec = GraphDecoder(layer, state)   # `state` becomes the static buffer set
        y = dec.step(x_t)                  # advances `state` in place
    """

    def __init__(self, layer, state, warmup=3):
        _check_capturable(state)
        self.layer = layer
        d = layer.cfg.d
        self.state = state
        ref = next(t for t in _leaves(state) if t.dim() >= 2)
        self.x = torch.zeros(ref.size(0), d, device=ref.device, dtype=ref.dtype)

        # Warm up on a side stream: allocator and any lazy init must not be
        # captured into the graph.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            probe = _clone_state(state)
            for _ in range(warmup):
                _, probe = layer.step(self.x, probe)
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            y, new = layer.step(self.x, self.state)
            self.y = y
            # write-back INSIDE the graph: one replay advances the recurrence, so
            # there is no per-token host-side state shuffling at all.
            for dst, src in _pairs(self.state, new):
                dst.copy_(src)

    def step(self, x_t):
        self.x.copy_(x_t)
        self.graph.replay()
        return self.y


def decode_all(layer, x, state=None, use_graph=True):
    """Token-by-token decode of x (B,T,d). Returns (y, state)."""
    B, T, _ = x.shape
    st = state if state is not None else layer.init_state(B, x.device, x.dtype)
    if use_graph and x.is_cuda:
        dec = GraphDecoder(layer, st)
        ys = [dec.step(x[:, t]).clone() for t in range(T)]
        return torch.stack(ys, 1), st
    ys = []
    for t in range(T):
        y, st = layer.step(x[:, t], st)
        ys.append(y)
    return torch.stack(ys, 1), st
