"""
SeqCond model — self-contained HuggingFace implementation.

All model code is embedded here so that trust_remote_code=True works without
any dependency on the original seqcond package.

Architecture:
  - Hybrid recurrent-transformer: every (seqcond_ratio+1)-th block (1-indexed)
    is a standard Transformer decoder block; the rest are SeqCond blocks.
  - SeqCond blocks use complex-exponential accumulators (den_acc, re_acc, im_acc)
    for O(1) per-token autoregressive decoding.
  - Transformer blocks use GQA with RoPE and KV-cache for autoregressive decoding.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_seqcond import SeqCondConfig

# ---------------------------------------------------------------------------
# Optional Triton kernels (accelerates SeqCond step, not required)
# ---------------------------------------------------------------------------
try:
    from .triton_kernels import (
        gated_rmsnorm_triton,
        seqcond_step_triton,
        TRITON_AVAILABLE,
    )
except ImportError:
    gated_rmsnorm_triton = None
    TRITON_AVAILABLE = False
    seqcond_step_triton = None


# ---------------------------------------------------------------------------
# Normalisation layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (x * self.scale.float()).to(orig)


class GatedRMSNorm(nn.Module):
    """RMSNorm with SiLU gating: rmsnorm(x * silu(residual))."""

    def __init__(self, hidden_size: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float() * F.silu(residual.float())
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return (x * self.weight.float()).to(orig)


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

def precompute_freqs(maxlen: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    half_d = head_dim // 2
    pos = np.arange(maxlen)[:, None]
    dim = np.arange(half_d)[None, :]
    angles = pos * (1.0 / (10000 ** (dim / half_d)))
    cos = torch.from_numpy(np.cos(angles).astype(np.float32))
    sin = torch.from_numpy(np.sin(angles).astype(np.float32))
    return cos, sin


def apply_rope(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = tensor.shape[-1] // 2
    cos = cos[..., :dim]
    sin = sin[..., :dim]
    x1, x2 = tensor[..., :dim], tensor[..., dim:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).view(tensor.shape)


# ---------------------------------------------------------------------------
# Transformer decoder block (GQA + RoPE)
# ---------------------------------------------------------------------------

class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self._num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_groups = num_heads // self._num_kv_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self._num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self._num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_groups == 1:
            return x
        b, l = x.shape[:2]
        extra = x.shape[2:]
        x = x.view(b, l, self._num_kv_heads, 1, *extra[1:])
        x = x.expand(b, l, self._num_kv_heads, self.num_groups, *extra[1:])
        return x.reshape(b, l, self.num_heads, *extra[1:])

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        b, l = x.shape[0], x.shape[1]
        q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, l, self._num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, l, self._num_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        cos_kv = cos[:, :, : self._num_kv_heads, :] if self._num_kv_heads < self.num_heads else cos
        sin_kv = sin[:, :, : self._num_kv_heads, :] if self._num_kv_heads < self.num_heads else sin
        k = apply_rope(k, cos_kv, sin_kv)

        if self.qk_norm:
            q_f = q.float(); k_f = k.float()
            q = (q_f * torch.rsqrt(q_f.pow(2).mean(-1, keepdim=True) + self.qk_norm_eps)).to(q.dtype)
            k = (k_f * torch.rsqrt(k_f.pow(2).mean(-1, keepdim=True) + self.qk_norm_eps)).to(k.dtype)

        k_cache = k; v_cache = v
        k = self._repeat_kv(k); v = self._repeat_kv(v)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("blhd,bmhd->bhlm", q, k) * scale
        causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = torch.where(causal, scores, torch.full_like(scores, -1e4))
        attn = F.softmax(scores.float(), dim=-1).to(v.dtype)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)
        out = torch.einsum("bhql,blhd->bqhd", attn, v).reshape(b, l, self.d_model).to(x.dtype)

        if return_state:
            return self.out_proj(out), (k_cache, v_cache)
        return self.out_proj(out)

    def step(
        self,
        x_t: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        pos: torch.Tensor,
        cos_t: torch.Tensor,
        sin_t: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        b = x_t.shape[0]
        q = self.q_proj(x_t).reshape(b, 1, self.num_heads, self.head_dim)
        k_new = self.k_proj(x_t).reshape(b, 1, self._num_kv_heads, self.head_dim)
        v_new = self.v_proj(x_t).reshape(b, 1, self._num_kv_heads, self.head_dim)

        q = apply_rope(q, cos_t, sin_t)
        cos_kv = cos_t[:, :, : self._num_kv_heads, :] if self._num_kv_heads < self.num_heads else cos_t
        sin_kv = sin_t[:, :, : self._num_kv_heads, :] if self._num_kv_heads < self.num_heads else sin_t
        k_new = apply_rope(k_new, cos_kv, sin_kv)

        if self.qk_norm:
            q_f = q.float(); k_f = k_new.float()
            q = (q_f * torch.rsqrt(q_f.pow(2).mean(-1, keepdim=True) + self.qk_norm_eps)).to(q.dtype)
            k_new = (k_f * torch.rsqrt(k_f.pow(2).mean(-1, keepdim=True) + self.qk_norm_eps)).to(k_new.dtype)

        k_cache, v_cache = kv_cache
        pos_idx = pos.long().view(b, 1, 1, 1).expand(-1, 1, k_new.size(2), k_new.size(3))
        k_cache.scatter_(1, pos_idx, k_new.to(k_cache.dtype))
        v_cache.scatter_(1, pos_idx, v_new.to(v_cache.dtype))

        if seq_len is not None:
            k_slice, v_slice = k_cache[:, :seq_len], v_cache[:, :seq_len]; L = seq_len
        else:
            k_slice, v_slice = k_cache, v_cache; L = k_cache.shape[1]

        k_r = self._repeat_kv(k_slice); v_r = self._repeat_kv(v_slice)
        mask = torch.arange(L, device=k_cache.device).view(1, 1, 1, L) > pos.long().view(b, 1, 1, 1)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k_r) * scale
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores.float(), dim=-1).to(v_r.dtype)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v_r).reshape(b, self.d_model).to(x_t.dtype)
        return self.out_proj(out), (k_cache, v_cache)


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, epsilon=norm_eps)
        self.attn = RotarySelfAttention(d_model, num_heads, num_kv_heads, dropout, qk_norm, qk_norm_eps)
        self.norm2 = RMSNorm(d_model, epsilon=norm_eps)
        self.ff_in = nn.Linear(d_model, 2 * d_ff, bias=True)
        self.ff_out = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x, cos, sin, mask=None, return_state=False):
        y = self.norm1(x)
        if return_state:
            y, kv = self.attn(y, cos=cos, sin=sin, mask=mask, return_state=True)
        else:
            y = self.attn(y, cos=cos, sin=sin, mask=mask)
        if self.dropout > 0 and self.training:
            y = F.dropout(y, p=self.dropout)
        x = x + y
        y = self.norm2(x)
        u, v = self.ff_in(y).chunk(2, dim=-1)
        y = self.ff_out(F.silu(v) * u)
        if self.dropout > 0 and self.training:
            y = F.dropout(y, p=self.dropout)
        out = x + y
        return (out, kv) if return_state else out

    def step(self, x_t, kv_cache, pos, cos_t, sin_t, seq_len=None):
        y = self.norm1(x_t)
        y, new_kv = self.attn.step(y, kv_cache, pos, cos_t, sin_t, seq_len=seq_len)
        x_t = x_t + y
        y = self.norm2(x_t)
        u, v = self.ff_in(y).chunk(2, dim=-1)
        return x_t + self.ff_out(F.silu(v) * u), new_kv


# ---------------------------------------------------------------------------
# SeqCond attention block
# ---------------------------------------------------------------------------

class SeqCondAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        num_query_heads: int = 6,
        num_anchor_heads: int = 0,
        num_thetas: int = 1,
        conv_kernel_size: int = 4,
        expand_factor: int = 1,
        out_expand_factor: int = 3,
        dropout: float = 0.0,
        maxlen: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert num_heads % num_query_heads == 0

        self.d_model = d_model
        self.K = num_heads
        self.K_q = num_query_heads
        self.n_rep = num_heads // num_query_heads
        self.M = num_thetas
        self.num_decay_heads = num_heads - num_anchor_heads
        self.num_anchor_heads = num_anchor_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout_rate = dropout
        self.maxlen = maxlen

        d_inner = int(d_model * expand_factor)
        self.H = max(1, d_inner // (self.K * self.M))
        self.dim_memory = self.K * self.H
        self.dim_query_head = self.H * self.M * 2
        self.dim_query_total = self.K_q * self.dim_query_head
        self.dim_expand = self.H * out_expand_factor
        self.dim_swiglu_head = self.dim_expand * 2
        self.dim_swiglu_total = self.K * self.dim_swiglu_head
        self.dim_mem_total = self.dim_memory + self.K
        self.dim_conv_total = self.dim_mem_total + self.dim_query_total

        self.in_proj = nn.Linear(d_model, self.dim_conv_total, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(self.dim_conv_total, 1, conv_kernel_size))
        nn.init.kaiming_normal_(self.conv_weight)

        # Cached buffers (computed lazily)
        self.register_buffer("_conv_kernel_t", None)
        self.register_buffer("_theta_cached", None)
        self.register_buffer("_w_int_cached", None)
        self.register_buffer("_decay_slopes_cached", None)
        self.register_buffer("_anchor_slopes_cached", None)
        self.register_buffer("_phase_scale_b", None)
        self.register_buffer("_score_scale_b", None)
        self.register_buffer("_score_bias_b", None)
        self._triton_out_re_buffer = None
        self._triton_out_im_buffer = None
        self._triton_norm_buffer = None

        if self.M == 1:
            init_theta = np.geomspace(0.001, 3.0, self.K).reshape(1, 1, self.K, 1, 1)
            init_theta = np.tile(init_theta, (1, 1, 1, self.H, 1))
            x = np.clip((init_theta - 0.001) / 2.999, 1e-4, 1 - 1e-4)
            self.theta_raw = nn.Parameter(torch.from_numpy((np.log(x) - np.log(1 - x)).astype(np.float32)))
            self.w_int_raw = nn.Parameter(torch.zeros(1, 1, self.K_q, self.n_rep, self.H, 1))
        else:
            init_vals = np.geomspace(0.001, 3.0, self.M).reshape(1, 1, 1, 1, self.M)
            init_vals = np.tile(init_vals, (1, 1, self.K, self.H, 1))
            self.theta_d_raw = nn.Parameter(torch.from_numpy(np.log(np.exp(init_vals) - 1.0 + 1e-4).astype(np.float32)))
            self.w_int_raw = nn.Parameter(torch.zeros(1, 1, self.K_q, self.n_rep, self.H, self.M))

        if self.num_decay_heads > 0:
            self.decay_slopes = nn.Parameter(
                torch.from_numpy(np.log(np.exp(np.geomspace(0.001, 0.1, self.num_decay_heads)) - 1).astype(np.float32))
            )
        if self.num_anchor_heads > 0:
            self.anchor_slopes = nn.Parameter(
                torch.from_numpy(np.log(np.exp(np.geomspace(0.01, 0.1, self.num_anchor_heads)) - 1).astype(np.float32))
            )

        self.score_scale = nn.Parameter(torch.ones(self.K))
        self.score_bias = nn.Parameter(torch.zeros(self.K))
        self.phase_scale = nn.Parameter(torch.ones(self.K))
        self.gate_proj = nn.Linear(d_model, self.K * 2 * self.H, bias=False)
        self.gated_norm = GatedRMSNorm(self.K * 2 * self.H)
        self.W_readout = nn.Parameter(torch.empty(self.K, 2 * self.H, self.dim_swiglu_head))
        nn.init.xavier_uniform_(self.W_readout)
        self.out_proj = nn.Linear(self.dim_swiglu_total // 2, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask=None, return_state: bool = False):
        B, L, D = x.shape
        z_conv = self.in_proj(x)
        z_conv_t = F.pad(z_conv.transpose(1, 2), (self.conv_kernel_size - 1, 0))
        z_conv = F.silu(F.conv1d(z_conv_t, self.conv_weight, groups=self.dim_conv_total).transpose(1, 2))

        z_mem = z_conv[..., : self.dim_mem_total]
        q_raw = z_conv[..., self.dim_mem_total :]
        k_val = z_mem[..., : self.dim_memory].reshape(B, L, self.K, self.H)
        s_raw = z_mem[..., self.dim_memory :]
        q_raw = q_raw.reshape(B, L, self.K_q, 1, self.H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        if self.M == 1:
            theta = 0.001 + 2.999 * torch.sigmoid(self.theta_raw)
        else:
            theta_d = F.softplus(self.theta_d_raw) + 1e-4
            theta_accum = torch.cumsum(theta_d, dim=-1)
            theta = 0.001 + (theta_accum / theta_accum[..., -1:]) * 2.999

        w_int = torch.exp(self.w_int_raw)
        w_int = w_int / (w_int.sum(dim=-1, keepdim=True) + 1e-6)

        pos = torch.arange(L, dtype=torch.float32, device=x.device)
        log_w_list = []
        if self.num_decay_heads > 0:
            slopes = F.softplus(self.decay_slopes).view(1, 1, -1)
            dist = torch.clamp((self.maxlen or L) - 1 - pos, min=0.0).view(1, L, 1)
            log_w_list.append(-slopes * dist)
        if self.num_anchor_heads > 0:
            log_w_list.append(-F.softplus(self.anchor_slopes).view(1, 1, -1) * pos.view(1, L, 1))
        log_tw = torch.cat(log_w_list, dim=2) if log_w_list else torch.zeros(1, L, self.K, device=x.device)

        score_raw = self.score_scale.view(1, 1, -1) * s_raw.float() + self.score_bias.view(1, 1, -1)
        p_w = (F.softplus(score_raw) * torch.exp(log_tw)).clamp(1e-4, 5000.0)

        k_f32 = k_val.float().unsqueeze(-1)
        p_w_b = p_w.unsqueeze(-1).unsqueeze(-1)
        phase_scale_b = self.phase_scale.view(1, 1, self.K, 1, 1)
        k_scaled = k_f32 * phase_scale_b
        phi = (k_scaled / (1.0 + k_scaled.abs())) * theta
        kvw = k_f32 * p_w_b
        re = kvw * torch.cos(phi)
        im = kvw * torch.sin(phi)

        flat_size = self.K * self.H * self.M
        stack = torch.cat([p_w.float(), re.reshape(B, L, -1), im.reshape(B, L, -1)], dim=-1)
        cumsum = torch.cumsum(stack, dim=1)
        den_acc = cumsum[..., : self.K]
        re_acc = cumsum[..., self.K : self.K + flat_size].reshape(B, L, self.K, self.H, self.M)
        im_acc = cumsum[..., self.K + flat_size :].reshape(B, L, self.K, self.H, self.M)

        inv_den = (1.0 / torch.clamp(den_acc, min=1e-4)).unsqueeze(-1).unsqueeze(-1)
        state_re_g = (re_acc * inv_den).reshape(B, L, self.K_q, self.n_rep, self.H, self.M)
        state_im_g = (im_acc * inv_den).reshape(B, L, self.K_q, self.n_rep, self.H, self.M)

        scale = 1.0 / (self.H ** 0.5)
        match_re = ((state_re_g * q_re + state_im_g * q_im) * scale).float()
        match_im = ((state_im_g * q_re - state_re_g * q_im) * scale).float()
        out_re = ((match_re * w_int.float()).sum(dim=-1)).reshape(B, L, self.K, self.H).to(x.dtype)
        out_im = ((match_im * w_int.float()).sum(dim=-1)).reshape(B, L, self.K, self.H).to(x.dtype)
        out_complex = self.gated_norm(torch.cat([out_re, out_im], dim=-1).reshape(B, L, -1), self.gate_proj(x))
        out_complex = out_complex.reshape(B, L, self.K, 2 * self.H)

        y_raw = torch.einsum("blkf,kfn->blkn", out_complex, self.W_readout.to(out_complex.dtype))
        y_val, y_gate = y_raw.chunk(2, dim=-1)
        output = self.out_proj((y_val * torch.sigmoid(y_gate)).reshape(B, L, -1).to(x.dtype))

        if return_state:
            z_pre = self.in_proj(x)
            buf_sz = self.conv_kernel_size - 1
            conv_buf = z_pre[:, -buf_sz:] if L >= buf_sz else torch.cat([
                torch.zeros(B, buf_sz - L, self.dim_conv_total, device=x.device, dtype=z_pre.dtype), z_pre], dim=1)
            state = (
                p_w.sum(dim=1),
                re_acc[:, -1],
                im_acc[:, -1],
                torch.full((B,), L, dtype=torch.float32, device=x.device),
                conv_buf,
            )
            return output, state
        return output

    def step(self, x_t: torch.Tensor, state: Tuple, use_triton: bool = False) -> Tuple:
        B, D = x_t.shape
        den_acc, re_acc, im_acc, pos, conv_buffer = state

        z_conv = self.in_proj(x_t)

        if self._conv_kernel_t is None or self._conv_kernel_t.device != z_conv.device:
            self._conv_kernel_t = self.conv_weight[:, 0, :].t().contiguous()

        conv_input = torch.cat([conv_buffer, z_conv.unsqueeze(1)], dim=1)
        z_conv_act = F.silu((conv_input * self._conv_kernel_t).sum(dim=1))

        z_mem = z_conv_act[..., : self.dim_mem_total]
        q_raw = z_conv_act[..., self.dim_mem_total :]
        k_val = z_mem[..., : self.dim_memory].reshape(B, self.K, self.H)
        s_raw = z_mem[..., self.dim_memory :]
        q_raw = q_raw.reshape(B, self.K_q, 1, self.H, self.M, 2)
        q_re, q_im = q_raw[..., 0], q_raw[..., 1]

        if self._theta_cached is None:
            if self.M == 1:
                self._theta_cached = (0.001 + 2.999 * torch.sigmoid(self.theta_raw))[0, 0]
            else:
                theta_d = F.softplus(self.theta_d_raw) + 1e-4
                theta_accum = torch.cumsum(theta_d, dim=-1)
                self._theta_cached = (0.001 + (theta_accum / theta_accum[..., -1:]) * 2.999)[0, 0]
            w = torch.exp(self.w_int_raw)
            self._w_int_cached = w / (w.sum(dim=-1, keepdim=True) + 1e-6)
            self._w_int_cached = self._w_int_cached[0, 0]
        theta = self._theta_cached
        w_int = self._w_int_cached

        if self._decay_slopes_cached is None and self.num_decay_heads > 0:
            self._decay_slopes_cached = F.softplus(self.decay_slopes).view(1, -1)
        if self._anchor_slopes_cached is None and self.num_anchor_heads > 0:
            self._anchor_slopes_cached = F.softplus(self.anchor_slopes).view(1, -1)
        if self._score_scale_b is None:
            self._score_scale_b = self.score_scale.view(1, -1)
            self._score_bias_b = self.score_bias.view(1, -1)
            self._phase_scale_b = self.phase_scale.view(1, self.K, 1, 1)

        log_w_list = []
        if self.num_decay_heads > 0:
            dist = (self.maxlen or 2048) - 1 - pos.unsqueeze(-1)
            log_w_list.append(-self._decay_slopes_cached * dist.clamp(min=0.0))
        if self.num_anchor_heads > 0:
            log_w_list.append(-self._anchor_slopes_cached * pos.unsqueeze(-1))
        log_tw = torch.cat(log_w_list, dim=1) if log_w_list else torch.zeros(B, self.K, device=x_t.device)

        if (
            use_triton
            and x_t.is_cuda
            and self.n_rep == 1
            and TRITON_AVAILABLE
            and seqcond_step_triton is not None
        ):
            if (
                self._triton_out_re_buffer is None
                or self._triton_out_re_buffer.shape != (B, self.K, self.H)
                or self._triton_out_re_buffer.device != x_t.device
            ):
                self._triton_out_re_buffer = torch.empty(
                    B, self.K, self.H, device=x_t.device, dtype=torch.float32
                )
                self._triton_out_im_buffer = torch.empty_like(
                    self._triton_out_re_buffer
                )
            out_re, out_im = seqcond_step_triton(
                k_val,
                s_raw,
                q_re.squeeze(2),
                q_im.squeeze(2),
                re_acc,
                im_acc,
                den_acc,
                theta,
                w_int,
                self.phase_scale,
                self.score_scale,
                self.score_bias,
                log_tw,
                out_re_buffer=self._triton_out_re_buffer,
                out_im_buffer=self._triton_out_im_buffer,
            )
            out_complex = torch.cat([out_re, out_im], dim=-1)
        else:
            score_raw = self._score_scale_b * s_raw.float() + self._score_bias_b
            p_w = (F.softplus(score_raw) * torch.exp(log_tw)).clamp(1e-4, 5000.0)
            k_f32 = k_val.float().unsqueeze(-1)
            k_scaled = k_f32 * self._phase_scale_b
            phi = (k_scaled / (1.0 + k_scaled.abs())) * theta
            kvw = k_f32 * p_w.unsqueeze(-1).unsqueeze(-1)
            re = kvw * torch.cos(phi)
            im = kvw * torch.sin(phi)
            den_acc.add_(p_w); re_acc.add_(re); im_acc.add_(im)
            inv_den = (1.0 / torch.clamp(den_acc, min=1e-4)).unsqueeze(-1).unsqueeze(-1)
            state_re_g = (re_acc * inv_den).reshape(B, self.K_q, self.n_rep, self.H, self.M)
            state_im_g = (im_acc * inv_den).reshape(B, self.K_q, self.n_rep, self.H, self.M)
            scale = 1.0 / (self.H ** 0.5)
            match_re = ((state_re_g * q_re + state_im_g * q_im) * scale).float()
            match_im = ((state_im_g * q_re - state_re_g * q_im) * scale).float()
            out_re = ((match_re * w_int.float()).sum(-1)).reshape(B, self.K, self.H).to(x_t.dtype)
            out_im = ((match_im * w_int.float()).sum(-1)).reshape(B, self.K, self.H).to(x_t.dtype)
            out_complex = torch.cat([out_re, out_im], dim=-1)

        out_complex = out_complex.reshape(B, self.K, 2 * self.H)
        out_complex_flat = out_complex.reshape(B, -1)
        gate_for_norm = self.gate_proj(x_t)
        if use_triton and x_t.is_cuda and gated_rmsnorm_triton is not None:
            if (
                self._triton_norm_buffer is None
                or self._triton_norm_buffer.shape != out_complex_flat.shape
                or self._triton_norm_buffer.device != x_t.device
            ):
                self._triton_norm_buffer = torch.empty(
                    out_complex_flat.shape,
                    device=x_t.device,
                    dtype=torch.float32,
                )
            out_flat = gated_rmsnorm_triton(
                out_complex_flat,
                gate_for_norm,
                self.gated_norm.weight,
                self.gated_norm.epsilon,
                out_buffer=self._triton_norm_buffer,
            )
        else:
            out_flat = self.gated_norm(out_complex_flat, gate_for_norm)
        out_complex = out_flat.to(x_t.dtype).reshape(B, self.K, 2 * self.H)
        y_raw = torch.einsum("bkf,kfn->bkn", out_complex, self.W_readout.to(out_complex.dtype))
        y_val, y_gate = y_raw.chunk(2, dim=-1)
        out = self.out_proj((y_val * torch.sigmoid(y_gate)).reshape(B, -1).to(x_t.dtype))

        pos.add_(1).clamp_(max=(self.maxlen or 2048) - 1)
        if self.conv_kernel_size > 1:
            if self.conv_kernel_size > 2:
                conv_buffer[:, :-1, :].copy_(conv_buffer[:, 1:, :].clone())
            conv_buffer[:, -1, :].copy_(z_conv)

        return out, (den_acc, re_acc, im_acc, pos, conv_buffer)


class SeqCondBlock(nn.Module):
    def __init__(self, d_model: int, norm_eps: float = 1e-6, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model, epsilon=norm_eps)
        self.attn = SeqCondAttention(d_model=d_model, **kwargs)

    def forward(self, x, mask=None, return_state=False):
        if return_state:
            out, state = self.attn(self.norm(x), mask=mask, return_state=True)
            return x + out, state
        return x + self.attn(self.norm(x), mask=mask)

    def step(self, x_t, state, use_triton=False):
        out, new_state = self.attn.step(self.norm(x_t), state, use_triton=use_triton)
        return x_t + out, new_state


# ---------------------------------------------------------------------------
# Core SeqCond language model
# ---------------------------------------------------------------------------

class SeqCondModel(nn.Module):
    """Core SeqCond model (no HF wrapper). Used internally by SeqCondForCausalLM."""

    def __init__(self, config: SeqCondConfig):
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.num_layers = config.num_layers
        self.vocab_size = config.vocab_size
        self.maxlen = config.maxlen
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self.seqcond_ratio = config.seqcond_ratio

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.use_positional_embedding = config.use_positional_embedding
        if config.use_positional_embedding:
            self.position_embedding = nn.Embedding(config.maxlen, config.d_model)

        head_dim = config.d_model // config.num_heads
        cos, sin = precompute_freqs(config.maxlen, head_dim)
        self.register_buffer("cos_emb", cos)
        self.register_buffer("sin_emb", sin)

        self.blocks = nn.ModuleList()
        self.block_types = []
        for i in range(config.num_layers):
            if (i + 1) % (config.seqcond_ratio + 1) == 0:
                block = TransformerDecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    num_kv_heads=self.num_kv_heads,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                )
                self.block_types.append("transformer")
            else:
                block = SeqCondBlock(
                    d_model=config.d_model,
                    num_heads=config.seqcond_heads,
                    num_query_heads=config.num_query_heads,
                    num_anchor_heads=config.num_anchor_heads,
                    num_thetas=config.num_thetas,
                    conv_kernel_size=config.conv_kernel_size,
                    expand_factor=config.expand_factor,
                    out_expand_factor=config.out_expand_factor,
                    dropout=config.dropout,
                    maxlen=config.maxlen,
                )
                self.block_types.append("seqcond")
            self.blocks.append(block)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.embedding(input_ids)
        if self.use_positional_embedding:
            x = x + self.position_embedding(torch.arange(L, device=input_ids.device))
        cos = self.cos_emb[:L].unsqueeze(0).unsqueeze(2).expand(B, L, self.num_heads, -1)
        sin = self.sin_emb[:L].unsqueeze(0).unsqueeze(2).expand(B, L, self.num_heads, -1)
        for block, bt in zip(self.blocks, self.block_types):
            x = block(x, cos, sin) if bt == "transformer" else block(x)
        return self.lm_head(x)

    def prefill(self, input_ids: torch.Tensor, return_all_logits: bool = False):
        B, L = input_ids.shape
        device = input_ids.device
        x = self.embedding(input_ids)
        if self.use_positional_embedding:
            x = x + self.position_embedding(torch.arange(L, device=device))
        cos = self.cos_emb[:L].unsqueeze(0).unsqueeze(2).expand(B, L, self.num_heads, -1)
        sin = self.sin_emb[:L].unsqueeze(0).unsqueeze(2).expand(B, L, self.num_heads, -1)
        states = []
        for block, bt in zip(self.blocks, self.block_types):
            if bt == "transformer":
                x, kv = block(x, cos, sin, return_state=True)
                k, v = kv
                k_cache = torch.zeros(B, self.maxlen, self.num_kv_heads, self.d_model // self.num_heads, device=device, dtype=k.dtype)
                v_cache = torch.zeros_like(k_cache)
                k_cache[:, :L] = k; v_cache[:, :L] = v
                states.append((k_cache, v_cache))
            else:
                x, state = block(x, return_state=True)
                states.append(state)
        logits = self.lm_head(x)
        if return_all_logits:
            return logits, states
        return logits[:, -1:, :], states

    def init_state(self, batch_size: int, device: torch.device) -> List:
        states = []
        for block, bt in zip(self.blocks, self.block_types):
            if bt == "transformer":
                k = torch.zeros(batch_size, self.maxlen, self.num_kv_heads, self.d_model // self.num_heads, device=device)
                states.append((k, torch.zeros_like(k)))
            else:
                a = block.attn
                states.append((
                    torch.zeros(batch_size, a.K, device=device),
                    torch.zeros(batch_size, a.K, a.H, a.M, device=device),
                    torch.zeros(batch_size, a.K, a.H, a.M, device=device),
                    torch.zeros(batch_size, device=device),
                    torch.zeros(batch_size, a.conv_kernel_size - 1, a.dim_conv_total, device=device),
                ))
        return states

    def step(self, token_id: torch.Tensor, states: List, pos=None, seq_len=None, use_triton=False):
        B = token_id.size(0)
        if pos is None:
            for state, bt in zip(states, self.block_types):
                if bt == "seqcond":
                    pos = state[3]; break
            if pos is None:
                pos = torch.zeros(B, device=token_id.device, dtype=torch.long)

        x = self.embedding(token_id).squeeze(1)
        pos = pos.clamp(max=self.maxlen - 1)
        if self.use_positional_embedding:
            x = x + torch.index_select(self.position_embedding.weight, 0, pos.long())

        pos_idx = pos.long()
        cos_t = torch.index_select(self.cos_emb, 0, pos_idx).unsqueeze(1).unsqueeze(1).expand(B, 1, self.num_heads, -1)
        sin_t = torch.index_select(self.sin_emb, 0, pos_idx).unsqueeze(1).unsqueeze(1).expand(B, 1, self.num_heads, -1)

        new_states = []
        for block, bt, state in zip(self.blocks, self.block_types, states):
            if bt == "transformer":
                x, ns = block.step(x, state, pos, cos_t, sin_t, seq_len=seq_len)
            else:
                x, ns = block.step(x, state, use_triton=use_triton)
            new_states.append(ns)

        return self.lm_head(x), new_states


# ---------------------------------------------------------------------------
# HuggingFace wrapper
# ---------------------------------------------------------------------------

class SeqCondPreTrainedModel(PreTrainedModel):
    config_class = SeqCondConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class SeqCondForCausalLM(SeqCondPreTrainedModel):
    """
    SeqCond causal language model, HuggingFace-compatible.

    Supports:
    - Standard HF forward() for training / perplexity evaluation.
    - Custom generate() using state-based O(1) decoding.
    - generate_batch() for batched generation with per-sample early stopping.
    - precompute() / use_cuda_graph=True for CUDA-graph-accelerated decoding.
    """

    _CUDA_GRAPH_SEQ_LENS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    def __init__(self, config: SeqCondConfig):
        super().__init__(config)
        self.model = SeqCondModel(config)
        self.post_init()
        # CUDA graph state
        self._cg_graphs: dict = {}
        self._cg_logits: dict = {}
        self._cg_token: Optional[torch.Tensor] = None
        self._cg_states: Optional[list] = None
        self._cg_use_triton: bool = False
        self._cg_ready: bool = False  # True after precompute() has been called

    # ------------------------------------------------------------------
    # CUDA graph helpers
    # ------------------------------------------------------------------

    def _cg_get_seq_len(self, pos: int) -> int:
        for s in self._CUDA_GRAPH_SEQ_LENS:
            if s >= pos + 1:
                return s
        return self._CUDA_GRAPH_SEQ_LENS[-1]

    def _cg_copy_states(self, src, dst):
        for s, d in zip(src, dst):
            for st, dt in zip(s, d):
                dt.copy_(st)

    def _cg_capture(self, seq_len: int):
        saved = self.model.init_state(1, device=self._cg_token.device)
        self._cg_copy_states(self._cg_states, saved)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.model.step(self._cg_token, self._cg_states,
                                seq_len=seq_len, use_triton=self._cg_use_triton)
        torch.cuda.current_stream().wait_stream(stream)
        self._cg_copy_states(saved, self._cg_states)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits, _ = self.model.step(self._cg_token, self._cg_states,
                                        seq_len=seq_len, use_triton=self._cg_use_triton)
        self._cg_copy_states(saved, self._cg_states)
        self._cg_graphs[seq_len] = graph
        self._cg_logits[seq_len] = logits

    @torch.no_grad()
    def precompute(self, max_seq_len: int = 2048, use_triton: bool = False):
        """Pre-capture CUDA graphs up to max_seq_len. Call once after loading."""
        if not torch.cuda.is_available():
            return
        if self._cg_use_triton != use_triton:
            self._cg_graphs = {}
            self._cg_logits = {}
        self._cg_use_triton = use_triton
        device = next(self.parameters()).device
        self._cg_token = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._cg_states = self.model.init_state(1, device=device)
        for s in self._CUDA_GRAPH_SEQ_LENS:
            if s > max_seq_len:
                break
            self._cg_capture(s)
        self._cg_ready = True
        print(f"Pre-captured {len(self._cg_graphs)} CUDA graphs (triton={use_triton}).")

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Standard forward pass (used for training / perplexity).

        Note: attention_mask is accepted for API compatibility but is not used
        in the forward pass — SeqCond is always causal.
        """
        logits = self.model(input_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    @staticmethod
    def _detect_triton() -> bool:
        try:
            import triton  # noqa: F401
            return True
        except ImportError:
            return False

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 1024,
        temperature: float = 0.15,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        acceleration: str = "auto",
        use_triton: Optional[bool] = None,
        use_cuda_graph: Optional[bool] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Autoregressive generation with state-based O(1) decoding.

        Args:
            acceleration: One of ``"auto"`` (default), ``"cuda_graph"``,
                ``"triton"`` (cuda_graph + triton), or ``"none"``.
                ``"auto"`` uses CUDA graphs when a GPU is available, and adds
                Triton kernels automatically if the triton package is installed.
                Explicit ``use_triton`` / ``use_cuda_graph`` kwargs override this.

        Returns the full sequence (prompt + generated tokens) as a LongTensor.
        """
        # ------------------------------------------------------------------
        # Resolve acceleration mode
        # ------------------------------------------------------------------
        on_cuda = torch.cuda.is_available() and input_ids.device.type == "cuda"

        if acceleration == "auto":
            _use_cuda_graph = on_cuda
            _use_triton     = on_cuda and self._detect_triton()
        elif acceleration == "triton":
            _use_cuda_graph = on_cuda
            _use_triton     = on_cuda
        elif acceleration == "cuda_graph":
            _use_cuda_graph = on_cuda
            _use_triton     = False
        else:  # "none"
            _use_cuda_graph = False
            _use_triton     = False

        # Legacy kwargs override
        if use_cuda_graph is not None:
            _use_cuda_graph = use_cuda_graph and on_cuda
        if use_triton is not None:
            _use_triton = use_triton and on_cuda

        # Lazy precompute on first generate() call
        if _use_cuda_graph and not self._cg_ready:
            self.precompute(max_seq_len=2048, use_triton=_use_triton)
        elif _use_cuda_graph and self._cg_use_triton != _use_triton:
            self.precompute(max_seq_len=2048, use_triton=_use_triton)

        use_triton     = _use_triton
        use_cuda_graph = _use_cuda_graph
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        device = input_ids.device
        B = input_ids.size(0)

        # Prefill
        logits, states = self.model.prefill(input_ids)
        logits = logits.squeeze(1)  # (B, vocab)

        generated = input_ids.tolist()
        finished = [False] * B
        token_buf = torch.zeros((B, 1), dtype=torch.long, device=device)
        seq_len = input_ids.size(1)

        # CUDA graph: sync prefill states into static buffer once before decode loop
        if use_cuda_graph and torch.cuda.is_available() and B == 1:
            if self._cg_token is None:
                self._cg_use_triton = use_triton
                self._cg_token = torch.zeros((1, 1), dtype=torch.long, device=device)
                self._cg_states = self.model.init_state(1, device=device)
            self._cg_copy_states(states, self._cg_states)
            states = self._cg_states

        for _ in range(max_new_tokens):
            # Temperature scaling
            if temperature > 0:
                ls = logits / temperature
            else:
                ls = logits.clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for bi, toks in enumerate(generated):
                    for t in set(toks):
                        if 0 <= t < self.config.vocab_size:
                            ls[bi, t] /= repetition_penalty

            # Sampling
            if temperature == 0:
                next_tokens = torch.argmax(ls, dim=-1)
            else:
                if top_k > 0:
                    kth = torch.topk(ls, top_k, dim=-1).values[:, -1:]
                    ls = ls.masked_fill(ls < kth, float("-inf"))
                if top_p < 1.0:
                    sorted_ls, sorted_idx = torch.sort(ls, dim=-1, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_ls, dim=-1), dim=-1)
                    sorted_remove = cum_probs > top_p
                    sorted_remove[:, 1:] = sorted_remove[:, :-1].clone()
                    sorted_remove[:, 0] = False
                    remove = torch.zeros_like(sorted_remove)
                    remove.scatter_(1, sorted_idx, sorted_remove)
                    ls = ls.masked_fill(remove, float("-inf"))
                probs = F.softmax(ls, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for bi in range(B):
                tok = next_tokens[bi].item()
                generated[bi].append(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    finished[bi] = True
                token_buf[bi, 0] = tok

            if all(finished):
                break

            seq_len += 1

            if use_cuda_graph and torch.cuda.is_available() and B == 1:
                cg_sl = self._cg_get_seq_len(seq_len - 1)
                if cg_sl not in self._cg_graphs:
                    self._cg_capture(cg_sl)
                self._cg_token.copy_(token_buf)
                self._cg_graphs[cg_sl].replay()
                logits = self._cg_logits[cg_sl]
            else:
                logits, states = self.model.step(token_buf, states, seq_len=seq_len, use_triton=use_triton)

        max_len = max(len(g) for g in generated)
        pad_id = self.config.pad_token_id or 0
        out = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        for bi, g in enumerate(generated):
            out[bi, : len(g)] = torch.tensor(g, dtype=torch.long, device=device)
        return out

    @torch.no_grad()
    def generate_batch(
        self,
        input_ids_list: List[torch.LongTensor],
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        eos_token_id: Optional[int] = None,
        use_triton: bool = False,
    ) -> List[List[int]]:
        """
        Batched generation: each prompt is prefilled independently, then
        decoded in lockstep with per-sample early stopping.

        Args:
            input_ids_list: List of 1D LongTensors, one per prompt.
        Returns:
            List of generated token id lists (completion only, EOS stripped).
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        device = input_ids_list[0].device
        B = len(input_ids_list)

        # Per-sample prefill
        all_logits, all_states = [], []
        for ids in input_ids_list:
            lg, st = self.model.prefill(ids.unsqueeze(0))
            all_logits.append(lg.squeeze(1))
            all_states.append(st)

        logits = torch.cat(all_logits, dim=0)
        # Stack states
        num_blocks = len(all_states[0])
        states = [
            tuple(torch.cat([s[i][j] for s in all_states], dim=0) for j in range(len(all_states[0][i])))
            for i in range(num_blocks)
        ]

        generated = [[] for _ in range(B)]
        finished = [False] * B
        active_map = list(range(B))
        token_buf = torch.zeros((B, 1), dtype=torch.long, device=device)
        seq_len = max(ids.size(0) for ids in input_ids_list)

        for _ in range(max_new_tokens):
            B_cur = len(active_map)
            if B_cur == 0:
                break

            if temperature == 0:
                next_tokens = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            newly_done = set()
            for bi in range(B_cur):
                oi = active_map[bi]
                tok = next_tokens[bi].item()
                generated[oi].append(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    finished[oi] = True
                    newly_done.add(bi)
                else:
                    token_buf[bi, 0] = tok

            if all(finished):
                break

            if newly_done:
                keep = [bi for bi in range(B_cur) if bi not in newly_done]
                if not keep:
                    break
                keep_idx = torch.tensor(keep, device=device)
                token_buf = token_buf[keep_idx].contiguous()
                states = [tuple(s[keep_idx].contiguous() for s in st) for st in states]
                active_map = [active_map[bi] for bi in keep]

            seq_len += 1
            logits, states = self.model.step(token_buf, states, seq_len=seq_len, use_triton=use_triton)

        results = []
        for toks in generated:
            if toks and toks[-1] == eos_token_id:
                toks = toks[:-1]
            results.append(toks)
        return results
