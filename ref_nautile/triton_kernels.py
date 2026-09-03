import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


if TRITON_AVAILABLE:
    def _select_seqcond_launch_config(H: int, M: int) -> tuple[int, int]:
        if M <= 1:
            block_m = 1
        elif M <= 2:
            block_m = 2
        elif M <= 4:
            block_m = 4
        elif M <= 8:
            block_m = 8
        else:
            block_m = 16

        if H >= 64:
            block_h = 64
        elif H >= 32:
            block_h = 32
        elif H >= 16:
            block_h = 16
        elif H >= 8:
            block_h = 8
        elif H >= 4:
            block_h = 4
        elif H >= 2:
            block_h = 2
        else:
            block_h = 1
        return block_m, block_h

    @triton.jit
    def _seqcond_fully_fused_kernel_impl(
        k_ptr,
        s_raw_ptr,
        q_re_ptr,
        q_im_ptr,
        re_acc_ptr,
        im_acc_ptr,
        den_acc_ptr,
        theta_ptr,
        w_int_ptr,
        phase_scale_ptr,
        score_scale_ptr,
        score_bias_ptr,
        log_tw_ptr,
        out_re_ptr,
        out_im_ptr,
        K: tl.constexpr,
        H: tl.constexpr,
        M: tl.constexpr,
        stride_k_b,
        stride_k_k,
        stride_k_h,
        stride_acc_b,
        stride_acc_k,
        stride_acc_h,
        stride_acc_m,
        stride_theta_k,
        stride_theta_h,
        stride_theta_m,
        stride_q_b,
        stride_q_k,
        stride_q_h,
        stride_q_m,
        stride_w_k,
        stride_w_h,
        stride_w_m,
        stride_out_b,
        stride_out_k,
        stride_out_h,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
        b = pid // (K * num_h_blocks)
        rem = pid % (K * num_h_blocks)
        k = rem // num_h_blocks
        h_block = rem % num_h_blocks
        h_start = h_block * BLOCK_H

        s_raw = tl.load(s_raw_ptr + b * K + k)
        score_scale = tl.load(score_scale_ptr + k)
        score_bias = tl.load(score_bias_ptr + k)
        log_tw = tl.load(log_tw_ptr + b * K + k)
        phase_scale = tl.load(phase_scale_ptr + k)

        score = score_scale * s_raw + score_bias
        p_w_content = tl.where(score > 20.0, score, tl.log(1.0 + tl.exp(score)))
        p_w = p_w_content * tl.exp(log_tw)
        p_w = tl.minimum(tl.maximum(p_w, 1e-4), 5000.0)

        old_den = tl.load(den_acc_ptr + b * K + k)
        new_den = old_den + p_w
        if h_block == 0:
            tl.store(den_acc_ptr + b * K + k, new_den)

        offs_h = tl.arange(0, BLOCK_H)
        h_idx = h_start + offs_h
        h_mask = h_idx < H
        k_val = tl.load(
            k_ptr + b * stride_k_b + k * stride_k_k + h_idx * stride_k_h,
            mask=h_mask,
            other=0.0,
        )
        k_scaled = k_val * phase_scale
        phi_base = k_scaled / (1.0 + tl.abs(k_scaled))
        kvw = k_val * p_w
        sum_re = tl.zeros((BLOCK_H,), dtype=tl.float32)
        sum_im = tl.zeros((BLOCK_H,), dtype=tl.float32)
        inv_den = 1.0 / tl.maximum(new_den, 1e-4)
        scale = 1.0 / tl.sqrt(float(H))
        offs_m = tl.arange(0, BLOCK_M)

        for m_start in range(0, M, BLOCK_M):
            m_idx = m_start + offs_m
            m_mask = m_idx < M
            theta_base = k * stride_theta_k
            theta_vals = tl.load(
                theta_ptr + theta_base + h_idx[:, None] * stride_theta_h + m_idx[None, :] * stride_theta_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            phi = phi_base[:, None] * theta_vals
            cos_phi = tl.cos(phi)
            sin_phi = tl.sin(phi)
            acc_base = b * stride_acc_b + k * stride_acc_k
            old_re = tl.load(
                re_acc_ptr + acc_base + h_idx[:, None] * stride_acc_h + m_idx[None, :] * stride_acc_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            old_im = tl.load(
                im_acc_ptr + acc_base + h_idx[:, None] * stride_acc_h + m_idx[None, :] * stride_acc_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            new_re = old_re + kvw[:, None] * cos_phi
            new_im = old_im + kvw[:, None] * sin_phi
            tl.store(
                re_acc_ptr + acc_base + h_idx[:, None] * stride_acc_h + m_idx[None, :] * stride_acc_m,
                new_re,
                mask=h_mask[:, None] & m_mask[None, :],
            )
            tl.store(
                im_acc_ptr + acc_base + h_idx[:, None] * stride_acc_h + m_idx[None, :] * stride_acc_m,
                new_im,
                mask=h_mask[:, None] & m_mask[None, :],
            )
            q_base = b * stride_q_b + k * stride_q_k
            q_re_vals = tl.load(
                q_re_ptr + q_base + h_idx[:, None] * stride_q_h + m_idx[None, :] * stride_q_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            q_im_vals = tl.load(
                q_im_ptr + q_base + h_idx[:, None] * stride_q_h + m_idx[None, :] * stride_q_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            w_base = k * stride_w_k
            w_vals = tl.load(
                w_int_ptr + w_base + h_idx[:, None] * stride_w_h + m_idx[None, :] * stride_w_m,
                mask=h_mask[:, None] & m_mask[None, :],
                other=0.0,
            )
            state_re = new_re * inv_den
            state_im = new_im * inv_den
            match_re = (state_re * q_re_vals + state_im * q_im_vals) * scale
            match_im = (state_im * q_re_vals - state_re * q_im_vals) * scale
            sum_re += tl.sum(match_re * w_vals, axis=1)
            sum_im += tl.sum(match_im * w_vals, axis=1)

        out_base = b * stride_out_b + k * stride_out_k
        tl.store(out_re_ptr + out_base + h_idx * stride_out_h, sum_re, mask=h_mask)
        tl.store(out_im_ptr + out_base + h_idx * stride_out_h, sum_im, mask=h_mask)


def seqcond_step_triton(
    k_val: torch.Tensor,
    s_raw: torch.Tensor,
    q_re: torch.Tensor,
    q_im: torch.Tensor,
    re_acc: torch.Tensor,
    im_acc: torch.Tensor,
    den_acc: torch.Tensor,
    theta: torch.Tensor,
    w_int: torch.Tensor,
    phase_scale: torch.Tensor,
    score_scale: torch.Tensor,
    score_bias: torch.Tensor,
    log_time_weight: torch.Tensor,
    out_re_buffer: torch.Tensor | None = None,
    out_im_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, K, H = k_val.shape
    M = theta.shape[2]
    K_q = q_re.shape[1]
    assert K_q == K, (
        f"Triton kernel requires n_rep==1 (K_q==K), got K_q={K_q}, K={K}. "
        f"Use PyTorch path for n_rep>1."
    )

    def _prep_f32(t: torch.Tensor) -> torch.Tensor:
        if t.dtype == torch.float32:
            return t
        return t.float()

    def _prep_f32_contiguous(t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.float32:
            t = t.float()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    k_val = _prep_f32(k_val)
    s_raw = _prep_f32_contiguous(s_raw)
    q_re = _prep_f32(q_re)
    q_im = _prep_f32(q_im)
    theta = _prep_f32(theta)
    phase_scale = _prep_f32_contiguous(phase_scale)
    score_scale = _prep_f32_contiguous(score_scale)
    score_bias = _prep_f32_contiguous(score_bias)
    log_time_weight = _prep_f32_contiguous(log_time_weight)
    if w_int.dim() == 4:
        w_int = w_int.squeeze(1)
    w_int = _prep_f32(w_int)

    if (
        out_re_buffer is None
        or out_re_buffer.shape != (B, K, H)
        or out_re_buffer.device != k_val.device
        or out_re_buffer.dtype != torch.float32
    ):
        out_re = torch.empty(B, K, H, device=k_val.device, dtype=torch.float32)
    else:
        out_re = out_re_buffer
    if (
        out_im_buffer is None
        or out_im_buffer.shape != (B, K, H)
        or out_im_buffer.device != k_val.device
        or out_im_buffer.dtype != torch.float32
    ):
        out_im = torch.empty(B, K, H, device=k_val.device, dtype=torch.float32)
    else:
        out_im = out_im_buffer

    common_args = (
        k_val,
        s_raw,
        q_re,
        q_im,
        re_acc,
        im_acc,
        den_acc,
        theta,
        w_int,
        phase_scale,
        score_scale,
        score_bias,
        log_time_weight,
        out_re,
        out_im,
        K,
        H,
        M,
        k_val.stride(0),
        k_val.stride(1),
        k_val.stride(2),
        re_acc.stride(0),
        re_acc.stride(1),
        re_acc.stride(2),
        re_acc.stride(3),
        theta.stride(0),
        theta.stride(1),
        theta.stride(2),
        q_re.stride(0),
        q_re.stride(1),
        q_re.stride(2),
        q_re.stride(3),
        w_int.stride(0),
        w_int.stride(1),
        w_int.stride(2),
        out_re.stride(0),
        out_re.stride(1),
        out_re.stride(2),
    )
    block_m, block_h = _select_seqcond_launch_config(H, M)
    grid = (B * K * ((H + block_h - 1) // block_h),)
    _seqcond_fully_fused_kernel_impl[grid](*common_args, BLOCK_M=block_m, BLOCK_H=block_h)
    return out_re, out_im


if TRITON_AVAILABLE:
    def _select_rmsnorm_block_size(n_cols: int) -> int:
        block = 1
        while block < n_cols:
            block *= 2
        return min(block, 4096)

    @triton.jit
    def _gated_rmsnorm_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        n_cols,
        stride_x_row,
        stride_residual_row,
        stride_out_row,
        epsilon,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row * stride_residual_row + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        gated = x * (residual * tl.sigmoid(residual))
        variance = tl.sum(gated * gated, axis=0) / n_cols
        inv_rms = tl.rsqrt(variance + epsilon)
        out = gated * inv_rms * weight
        tl.store(out_ptr + row * stride_out_row + offs, out, mask=mask)


def gated_rmsnorm_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    out_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if x.dim() != 2 or residual.dim() != 2:
        raise ValueError(
            f"gated_rmsnorm_triton expects 2D tensors, got x.shape={tuple(x.shape)} residual.shape={tuple(residual.shape)}"
        )
    if x.shape != residual.shape:
        raise ValueError(
            f"gated_rmsnorm_triton expects matching x/residual shapes, got {tuple(x.shape)} and {tuple(residual.shape)}"
        )
    if weight.dim() != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError(
            f"gated_rmsnorm_triton expects weight.shape == ({x.shape[1]},), got {tuple(weight.shape)}"
        )

    def _prep_f32_contiguous(t: torch.Tensor) -> torch.Tensor:
        if t.dtype != torch.float32:
            t = t.float()
        if not t.is_contiguous():
            t = t.contiguous()
        return t

    x = _prep_f32_contiguous(x)
    residual = _prep_f32_contiguous(residual)
    weight = _prep_f32_contiguous(weight)
    rows, n_cols = x.shape
    if (
        out_buffer is None
        or out_buffer.shape != x.shape
        or out_buffer.device != x.device
        or out_buffer.dtype != torch.float32
    ):
        out = torch.empty_like(x, dtype=torch.float32)
    else:
        out = out_buffer
    block_n = _select_rmsnorm_block_size(n_cols)
    _gated_rmsnorm_kernel[(rows,)](
        x,
        residual,
        weight,
        out,
        n_cols,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        epsilon,
        BLOCK_N=block_n,
    )
    return out
