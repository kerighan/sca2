# SCA2 — complete mathematical specification

Every formula here is the one the code implements, and every equivalence claimed
is checked numerically in float64 at 1e-10 by `python -m sca2.iso` (prefill,
token-by-token decode, prefill-then-decode continuation, CUDA-graph decode, and
all gradients).

## Notation

| symbol | meaning | default |
|---|---|---|
| `B, T, d` | batch, sequence length, model width | 8, 128, 128 |
| `dv` | value width of both heads' state | `d/2` = 64 |
| `Mc` | C-head frequency channels | 64 |
| `Md` | D-head state channels | 16 |
| `G`, `gs` | D-head gate groups over `dv`, and `gs = dv/G` | 8, 8 |
| `b, t, s` | batch, query position, key position | |
| `m` | channel index (`Mc` or `Md`) | |
| `j` | value index in `[0, dv)`; `g = j / gs` its group | |

`z = LayerNorm(x)` and `h` is `z` shifted right by one, i.e. `h[:,0] = z_prev`
(the carried state, zero at sequence start) and `h[:,t] = z[:,t-1]`.

RMS normalization, applied per token at each head's output:

```
rms(u) = u / sqrt(mean_k u_k^2 + 1e-6)
```

---

## 1. The layer

```
z    = LayerNorm(x)
h    = shift(z)
u_c  = CHead(z, h)                      (B,T,2dv)
u_d  = DHead(z, h)                      (B,T,2dv)
x    = x + Mix([u_c ; u_d])             Mix: 4dv -> d
x    = x + FFN(LayerNorm(x))            FFN:  d -> ff -> d, GELU
```

Everything is per-token except the two heads, which carry state along `t`. Both
heads are exactly recurrent, so the layer decodes one token at a time in O(1)
(verified: 3e-16 in float64).

---

## 2. C head

### 2.1 Definition

Projections `K: d -> Mc`, `V: d -> dv` (no bias); parameters `theta, wr, wi` in
`R^Mc`; a fixed frequency grid `omega in R^Mc` (section 2.4).

Phases, with `p` the absolute position:

```
pw[b,t,m] = K(h)[b,t,m] . theta[m] + p_t . omega[m]        "write" phase
pq[b,t,m] = K(z)[b,t,m] . theta[m] + p_t . omega[m]        "read"  phase
v [b,t,j] = V(z)[b,t,j]
```

State — a running sum over positions, one complex accumulator per `(m, j)`:

```
S[b,t,m,j] = sum_{s<=t} v[b,s,j] . exp(i . pw[b,s,m])
```

Read-out, rotating the state by the conjugate read phase and mixing the `Mc`
channels with the complex weight `w = wr + i.wi`:

```
R[b,t,m,j] = S[b,t,m,j] . exp(-i . pq[b,t,m])
u          = rms( concat( Re[ mean_m R.w ] , Im[ mean_m R.w ] ) )    (B,T,2dv)
```

### 2.2 The key identity: the score is a scalar

Substituting `S` into `R` and taking the mean over `m`:

```
mean_m R[b,t,m,j] . w[m]
  = (1/Mc) sum_m w[m] sum_{s<=t} v[b,s,j] . exp(i(pw[b,s,m] - pq[b,t,m]))
  = sum_{s<=t} v[b,s,j] . kappa[b,t,s]
```

with

```
kappa[b,t,s] = (1/Mc) sum_m (wr[m] + i.wi[m]) . exp(i(pw[b,s,m] - pq[b,t,m]))
```

**`kappa` carries no `j` index.** The contraction over `m` is independent of the
value dimension, so the C head is *exactly* causal attention with a structured
complex score — a `B x T x T` scalar matrix, not a `B x T x Mc x dv` tensor.

### 2.3 Fast form (v1)

Expanding `cos(pw - pq)` and `sin(pw - pq)` turns both parts of `kappa` into one
real matmul over a `2Mc`-dimensional feature map. With
`cw = cos(pw), sw = sin(pw), cq = cos(pq), sq = sin(pq)`:

```
A  = wr.cw - wi.sw                       Bm = wr.sw + wi.cw
Fq = [cq , sq]                           (B,T,2Mc)
Kre = Fq @ [A , Bm]^T  / Mc              Kim = Fq @ [Bm , -A]^T / Mc
u   = rms( concat( tril(Kre) @ v , tril(Kim) @ v ) )
```

Cost `B.T^2.(2Mc + dv)` instead of `B.T.Mc.dv`, but six retained
`B.T.Mc.dv` activations collapse to two `B.T.T` score matrices. Measured: head
forward 35.9 -> 0.65 ms, memory 162 -> 25 MB at `B=8, T=128`.

**Carried state.** With a non-empty incoming state `S0`, write
`c1 = wr.cq + wi.sq` and `c2 = wr.sq - wi.cq`; the contribution is

```
u_re += (1/Mc) ( <c1, Re S0> + <c2, Im S0> )        contracted over m
u_im += (1/Mc) ( <c1, Im S0> - <c2, Re S0> )
```

and the closing state is `S[:, -1]`, i.e. `sum_t v ⊗ exp(i.pw)` plus the
incoming one.

### 2.4 The frequency grid, and what it does at init

At initialization (`theta = 0`, `wr = 1`, `wi = 0`) the phases lose their content
dependence and `kappa` collapses to a function of the lag `n = s - t` alone:

```
kappa[n] = (1/Mc) sum_m cos(n . omega[m])
```

a **Dirichlet kernel**. So the grid decides what delta the head starts as, and
the off-peak mass is conserved at 1.0 for any equispaced grid — only its shape
changes.

| grid | `omega[m]` | `kappa[0]` | `kappa[1]` | `kappa[Mc]` | off-peak mass |
|---|---|---|---|---|---|
| `dft` | `2.pi.m / Mc` | 1 | 0 | **1** | 1.0 |
| `len` | `2.pi.m / L` | 1 | 0.016 | 0 | 1.0 |
| `rope` | `pi . base^(-m/(Mc-1))` | 1 | 0.808 | 0.311 | **50.2** |

`dft` makes the geometric sum exact — `sum_m exp(i.2.pi.m.n/Mc) = Mc.delta(n mod
Mc)` — so the head is a *perfect* delta, at the price of putting the entire
sidelobe budget into one alias spike at lag `Mc`. `Mc` is therefore literally the
number of positions the head can address: the `[cos|sin]` basis has rank `Mc`.

Measured val loss (3 seeds, 1500 steps, compact vocab): `dft` 0.8235, `len`
0.7666, `rope` **0.7102**. The delta-at-init argument does not predict trained
quality.

### 2.5 Chunked form (v3)

The state carry is exact, so the quadratic form can be applied per chunk of
length `C`, with the state crossing chunk boundaries. Memory falls from `B.T.T`
to `B.C.C`, cost from `B.T^2.(2Mc+dv)` to `B.T.C.(2Mc+dv) + 2.B.T.Mc.dv`. At
`T <= C` this *is* v1, bit for bit.

### 2.6 Decode

```
S  <- S + v_t ⊗ exp(i.pw_t)
u  = rms( concat( Re[mean_m S.exp(-i.pq_t).w] , Im[...] ) )
```

O(1) per token, state `(B, Mc, dv)` — fixed size. The absolute position must
live in the state as a **tensor**: a host-side int is frozen by CUDA-graph
capture, making every replayed token decode at position 0 (0.52 relative error,
invisible to an eager test).

---

## 3. D head

### 3.1 Definition

Projections `V: d -> dv`, `gr, gi: d -> Md.G` (with bias), and the query
`qr, qi: d -> Md.dv`.

Gate (complex, contractive by construction, `|a| <= 1`):

```
a[b,t,m,g] = ( tanh(gr(h)) + i . tanh(gi(h)) )[b,t,m,g] / sqrt(2)
```

Recurrence — a diagonal complex linear scan with a **real** input, broadcast over
all `Md` channels (`g = g(j)` is `j`'s group):

```
S[b,t,m,j] = a[b,t,m,g] . S[b,t-1,m,j] + v[b,t,j]
```

Note the input enters only the real part, and `v` does not depend on `m`.

Query, normalized per element, and the read-out:

```
q[b,t,m,j] = (qr + i.qi)[b,t,m,j] / |(qr + i.qi)[b,t,m,j]|
u = rms( concat( Re[mean_m S.conj(q)] , Im[mean_m S.conj(q)] ) )
```

**No scalar-kernel reduction exists here**: `q` depends on `j`, so the
`(t, m, j)` state genuinely has to be materialized. `B.T.Md.dv` is the floor.

### 3.2 Fast form (v1): chunked log-space scan

Over a chunk of length `C`, the recurrence unrolls in closed form:

```
S[t] = A[t] . S_in + sum_{r<=t} D[t,r] . v[r]
A[t] = prod_{u<=t} a[u]              D[t,r] = prod_{r<u<=t} a[u]
```

Both products are built in log-magnitude / phase space, so they become cumsum
differences. With `la = log|a|` and `ph = arg(a)`, and `cla, cph` their inclusive
cumsums within the chunk:

```
D[t,r] = exp(cla[t] - cla[r]) . exp(i(cph[t] - cph[r])),   r <= t
A[t]   = exp(cla[t]) . exp(i.cph[t])
```

`la <= 0`, so `cla` is non-increasing and the exponent is `<= 0` on every kept
entry: nothing overflows, and — unlike the naive `A[t]/A[r]` — nothing is divided
by a decayed prefix. Gradients stay bounded because `d(la)/d(gr) = gr/|a|^2` is
always multiplied by a `D` factor carrying `|a[u]|`.

> **Mask before `exp`, not after.** On *masked* entries (`r > t`) the exponent can
> be large and positive, and `inf * 0 = NaN`. A zero-padded chunk makes this
> reachable: a padded gate gives `|a|^2 = tiny`, hence `la = -354` in float64 and
> a masked exponent of `+2620`. Clamp the exponent to `max=0` before `exp`.

The intra-chunk term is then a matmul per `(b, m, g)`: `(C x C) @ (C x gs)`.

### 3.3 Two-level (loop-free) form — measured slower

The chunk carry `carry[n+1] = A_last[n] . carry[n] + S_end[n]` is the same
recurrence one level up, so it too has a closed form and the python loop can be
removed entirely. Measured **2.6x slower** (6.31 vs 2.43 ms): the loop was doing
cache blocking. Kept as `x_loopfree`.

### 3.4 Decode

```
S <- a_t . S + v_t                (v_t real, added to the real part)
u  = rms( concat( Re[mean_m S.conj(q_t)] , Im[...] ) )
```

State `(B, Md, dv)`, fixed size.

---

## 4. sepq — separable D-head query

The only change: `q` is factored as an outer product.

### 4.1 Motivation

`qr, qi: d -> Md.dv` are **53.9% of the layer's MACs and 62.6% of its
parameters**, and they give the query `Md.dv = 1024` degrees of freedom per
token — exactly the size of the state it reads. But the state's `m`-dependence
comes *only* from the gate phase (`v` is shared across `m`), so those degrees of
freedom cannot be used independently.

### 4.2 Definition

Replace the two `d -> Md.dv` maps by four small ones:

```
alpha[b,t,m] = (qa_r + i.qa_i)(z)[b,t,m]        qa_r, qa_i : d -> Md
beta [b,t,j] = (qb_r + i.qb_i)(z)[b,t,j]        qb_r, qb_i : d -> dv
q[b,t,m,j]   = alpha[b,t,m] . beta[b,t,j]
```

The per-element normalization **factors exactly**, because
`|alpha.beta| = |alpha|.|beta|`: normalize `alpha` over `m` and `beta` over `j`
separately and their product is already unit-modulus. No approximation is
involved in the normalization.

### 4.3 The read-out collapses

Writing `ar, ai` and `br, bi` for the normalized real/imaginary parts:

```
q_re = ar.br - ai.bi            q_im = ar.bi + ai.br

u_re = mean_m ( S_re.q_re + S_im.q_im )
     = br . mean_m(S_re.ar + S_im.ai)  +  bi . mean_m(S_im.ar - S_re.ai)
u_im = mean_m ( S_im.q_re - S_re.q_im )
     = br . mean_m(S_im.ar - S_re.ai)  -  bi . mean_m(S_re.ar + S_im.ai)
```

So with

```
P[b,t,j] = mean_m ( S_re.ar + S_im.ai )        Q[b,t,j] = mean_m ( S_im.ar - S_re.ai )
u = rms( concat( br.P + bi.Q , br.Q - bi.P ) )
```

`m` is contracted **before** the query is applied. Consequences:

* no `(B,T,Md,dv)` query tensor is ever materialized — which is what made every
  earlier `q` experiment regress on the backward;
* the contraction is matmul-shaped rather than elementwise-then-mean.

### 4.4 Cost

At `d=128, Md=16, dv=64`:

| | full `q` | separable `q` |
|---|---|---|
| parameters | 262 144 | 20 480 |
| MACs | 268 M | 21 M |
| layer parameters | 419 264 | 177 600 |
| layer fwd+bwd (`B=8, T=128`) | 10.05 ms | **7.92 ms** |

Everything else — gate, recurrence, chunked scan, state shape, decode — is
unchanged, so the O(1) decode and its cost are identical to v1 (measured: 285 vs
284 us/token graphed, flat from `L=128` to `L=131072`).

### 4.5 What it costs in quality

val loss, 2000 steps, compact vocab, best of two learning rates:

| arm | params | `dft` | `rope` |
|---|---|---|---|
| full `Md=16` | 419k | 0.7735 | 0.6576 |
| sepq `Md=16` | 177k | 0.8183 | 0.6919 |
| sepq `Md=64` | 289k | 0.7839 | 0.6672 |
| sepq `Md=128` | 437k | 0.7425 | 0.6475 |

So it is a **reallocation, not a free saving**: at matched budget (`Md=128`,
437k vs 419k) it wins, at `Md=16` it costs +0.034..0.045. But `Md` is
super-linear in time (decay matrices are `B.Md.G.C.T`, state is `Md.dv`), so
`Md=128` is 4.4x slower — `Md=16` is the only point on the curve that buys time.
For scale, a transformer at the same 177k reaches 1.0037.

---

## 4b. Log-polar gate

The cartesian parameterization computes the gate with two `tanh`, then
immediately takes `log|a|` and `arg(a)` to run the scan — reconstructing a polar
form the network could emit directly:

```
cartesian:  a = (tanh(w_r) + i.tanh(w_i)) / sqrt(2)
            log|a| = 0.5 log(w_r^2 + w_i^2 + tiny)      arg = atan2(w_i, w_r)
polar:      log|a| = -softplus(w_mag)                   arg = w_phase
```

Same parameter shapes and count (two `Linear(d, Md.G)` with bias), so it is a
reinterpretation, not a resize — but a different function, hence an A/B.

Measured (1500 steps, `rope`, compact vocab): 0.7471 cartesian vs **0.7462**
polar, i.e. −0.0009 against a seed spread of ~0.02. Speed: 7.29 vs 7.16 us/token
at `B=16`, inside a 13–21% spread. **Neutral on both axes.**

It is nonetheless the form to prefer, for three numerical reasons:

* `|a| <= 1` holds *by construction*, not as a consequence of `tanh/sqrt(2)`
  arithmetic;
* the `tiny` floor disappears, and with it the NaN class of section 3.2 — a
  zero-padded gate gives `log|a| = -softplus(0) = -0.693` instead of `-354`, so
  a masked exponent can no longer reach `+2620` and the `clamp(max=0)` guard
  becomes unnecessary;
* the gradient becomes `d(log|a|)/dw = -sigmoid(w)`, bounded, instead of
  `w_r/|a|^2`, which diverges as `|a| -> 0`.

That matters most right before hand-writing a kernel: there is no reason to
carve an anti-NaN workaround into CUDA.

---

## 5. Complexity summary

| | prefill | decode / token | state | notes |
|---|---|---|---|---|
| C head (v1) | `B.T^2.(2Mc+dv)` | `O(Mc.dv)` | `B.Mc.dv` | memory `B.T.T` |
| C head (v3) | `B.T.C.(2Mc+dv)` | `O(Mc.dv)` | `B.Mc.dv` | memory `B.C.C` |
| D head | `B.T.Md.C.dv` | `O(Md.dv)` | `B.Md.dv` | decay memory `B.Md.G.C.T` |
| attention | `B.T^2.d` | `O(L.d)` | `2.B.L.d` | KV cache grows |

Decode is O(1) in context length for both heads and measured flat to 131 072
tokens; state is 0.33 MB constant against 1074 MB of KV cache at that length.
Prefill is the binding constraint, which is what section 2.5 addresses.
