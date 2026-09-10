"""Stacking: a language model made of any layers that speak the (prefill, step,
init_state) protocol -- LaplaceAttention, or a baseline from lapa.benchmarks.

    LM(layer_factory, n_layers, vocab, d)     generic
    LaplaceLM(vocab, d, n_layers, **cfg)      LaplaceAttention with LaplaceConfig(d=d, **cfg)

forward(tokens) -> logits              parallel, for training
prefill(tokens, state) -> logits, state
step(token, state) -> logits, state    one token, O(1) in the context length
generate(prompt, n, ...) -> ids        prefill the prompt, then step
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import LaplaceAttention, LaplaceConfig


class LM(nn.Module):
    def __init__(
        self,
        layer_factory: Callable[[int], nn.Module],
        n_layers: int,
        vocab: int,
        d: int,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.d, self.vocab = d, vocab
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([layer_factory(i) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        if tie_embeddings:
            self.head.weight = self.embed.weight

    # ---- protocol ---------------------------------------------------------- #
    def init_state(self, B: int, device) -> List[Dict]:
        return [layer.init_state(B, device) for layer in self.layers]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.prefill(tokens)[0]

    def prefill(self, tokens: torch.Tensor, state: Optional[List[Dict]] = None):
        B = tokens.size(0)
        st = state if state is not None else self.init_state(B, tokens.device)
        x = self.embed(tokens)
        new = []
        for layer, s in zip(self.layers, st):
            x, s = layer.prefill(x, s)
            new.append(s)
        return self.head(self.norm(x)), new

    def step(self, token: torch.Tensor, state: List[Dict]):
        """token (B,) -> logits (B,V)."""
        x = self.embed(token)
        new = []
        for layer, s in zip(self.layers, state):
            x, s = layer.step(x, s)
            new.append(s)
        return self.head(self.norm(x)), new

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        n: int,
        temperature: float = 1.0,
        top_k: int = 0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """prompt (B,T) or (T,) of ids -> (B,T+n) or (T+n,). Constant memory per step."""
        squeeze = prompt.dim() == 1
        ids = prompt[None] if squeeze else prompt
        logits, st = self.prefill(ids)
        out = [ids]
        nxt = logits[:, -1]
        for _ in range(n):
            if greedy or temperature <= 0:
                tok = nxt.argmax(-1)
            else:
                lg = nxt / temperature
                if top_k:
                    v, _ = lg.topk(min(top_k, lg.size(-1)))
                    lg = lg.masked_fill(lg < v[:, -1:], float("-inf"))
                tok = torch.multinomial(F.softmax(lg, -1), 1)[:, 0]
            out.append(tok[:, None])
            nxt, st = self.step(tok, st)
        res = torch.cat(out, 1)
        return res[0] if squeeze else res

    # ---- bookkeeping ------------------------------------------------------- #
    def layer_params(self) -> int:
        return sum(p.numel() for p in self.layers.parameters())

    def state_floats(self) -> Optional[int]:
        """Decode state per sequence, all layers; None if a layer has no fixed-size state."""
        tot = 0
        for layer in self.layers:
            f = getattr(layer, "state_floats", None)
            if f is None:
                return None
            tot += f()
        return tot


class LaplaceLM(LM):
    def __init__(
        self,
        vocab: int,
        d: int = 128,
        n_layers: int = 4,
        tie_embeddings: bool = False,
        **cfg,
    ):
        self.cfg = LaplaceConfig(d=d, **cfg)
        super().__init__(
            lambda i: LaplaceAttention(self.cfg), n_layers, vocab, d, tie_embeddings
        )
