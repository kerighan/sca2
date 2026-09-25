"""Two C heads, one token at a time, in numpy. One sequence (no batch dim).

  LONG head  = the current cdelta C head: rope grid, accumulator S, error-correcting write.
  SHORT head = the proposed dft head: Fourier grid of period L, SLIDING SUM S kept
               by adding the new write and subtracting the write from L steps ago.

Both share the same read and the same kernel kappa(t,s); only the grid and the
window differ. Each is checked against its closed form.  python chead_numpy.py
"""
import numpy as np

rng = np.random.default_rng(0)
d, dv, T = 8, 4, 40                                  # model width, value width, sequence length
sig = lambda x: 1 / (1 + np.exp(-x))

# ---- inputs: z_t = normalised token t, h_t = z_{t-1} (the write key uses the PREVIOUS token)
Z = rng.normal(size=(T, d))                          # (T, d)
H = np.vstack([np.zeros((1, d)), Z[:-1]])            # (T, d)


def make_params(M):
    return dict(
        K=rng.normal(size=(d, M)) / np.sqrt(d),      # key projection      (d, M)
        V=rng.normal(size=(d, dv)) / np.sqrt(d),     # value projection    (d, dv)
        theta=0.3 * rng.normal(size=M),              # content phase scale (M,)
        w=np.ones(M) + 0j,                           # spectral read weights, complex (M,)
        b=rng.normal(size=d) / np.sqrt(d), b0=-2.0,  # erase gate beta = sigmoid(b.z + b0)
    )


def kernel(p, omega, t, s, zt, hs):
    """kappa(t,s) = (1/M) sum_m w_m exp(i[theta_m (K(h_s) - K(z_t))_m + (s - t) omega_m])  -> scalar"""
    phi = hs @ p["K"] * p["theta"] + s * omega         # write phase of token s   (M,)
    psi = zt @ p["K"] * p["theta"] + t * omega         # read  phase of token t   (M,)
    return (p["w"] * np.exp(1j * (phi - psi))).sum() / len(omega)


# =============================================================================
# LONG HEAD: rope grid, accumulator, delta-rule write            (== cdelta)
# =============================================================================
M = 16
omega_long = 10000.0 ** (-np.arange(M) / M)          # rope grid: no aliasing to ~1e4 positions (M,)
p = make_params(M)

S = np.zeros((M, dv), complex)                       # accumulator state        (M, dv)
E = []                                               # written values, for the closed-form check
out_long = np.zeros((T, 2 * dv))                     # output per token         (T, 2dv)  [Re | Im]

for t in range(T):
    z, h = Z[t], H[t]                                # (d,), (d,)
    phi = h @ p["K"] * p["theta"] + t * omega_long   # write phase              (M,)
    c = np.exp(1j * phi)                             # write code, |c_m| = 1    (M,)
    v = z @ p["V"]                                   # value                    (dv,)

    # --- delta rule: read what is already stored at THIS code, write the error
    vhat = (np.conj(c) @ S).real / M                 # stored value at code c   (dv,)   (||c||^2 = M exactly)
    beta = sig(p["b"] @ z + p["b0"])                 # erase gate               scalar in (0,1)
    e = v - beta * vhat                              # written value            (dv,)
    S = S + np.outer(c, e)                           # S <- S + c (x) e         (M, dv)
    E.append(e)

    # --- read with the CURRENT token's phase
    psi = z @ p["K"] * p["theta"] + t * omega_long   # read phase               (M,)
    q = np.exp(-1j * psi)                            # read code                (M,)
    o = (p["w"] * q) @ S / M                         # complex read             (dv,)
    out_long[t] = np.concatenate([o.real, o.imag])   # (2dv,)   (the real layer RMS-normalises here)

# closed form: o_t = sum_{s<=t} kappa(t,s) e_s
t = T - 1
ref = sum(kernel(p, omega_long, t, s, Z[t], H[s]) * E[s] for s in range(t + 1))
assert np.allclose(out_long[t], np.concatenate([ref.real, ref.imag])), "long head != closed form"
print(f"LONG  head: state S {S.shape} complex = {2*S.size} floats, "
      f"grows with nothing (O(1) in T); closed form OK")

# =============================================================================
# SHORT HEAD: dft grid of period L, additive write, SLIDING SUM over the last L tokens
# =============================================================================
L = 8
omega_short = 2 * np.pi * np.arange(L) / L           # dft grid: (1/L) sum_m e^{i n omega_m} = delta(n mod L)   (L,)
p2 = make_params(L)
p2["theta"][:] = 0.0                                 # start content-free to SEE the Dirichlet delta; set != 0 for content taps

S2 = np.zeros((L, dv), complex)                      # sliding sum              (L, dv)
buf = []                                             # ring buffer of the last L writes (c_s, e_s); this IS the state
out_short = np.zeros((T, 2 * dv))

for t in range(T):
    z, h = Z[t], H[t]
    phi = h @ p2["K"] * p2["theta"] + t * omega_short  # write phase            (L,)
    c = np.exp(1j * phi)                               # write code             (L,)
    e = z @ p2["V"]                                    # additive: e = v, nothing accumulates so no delta rule   (dv,)

    S2 = S2 + np.outer(c, e)                           # add the new write      (L, dv)
    buf.append((c, e))
    if len(buf) > L:                                   # window full: SUBTRACT the write from L steps ago
        c_old, e_old = buf.pop(0)
        S2 = S2 - np.outer(c_old, e_old)               # exact: S2 = sum_{s=t-L+1}^{t} c_s (x) e_s
    # (len(buf) <= L: the state is L codes + L values, a fixed 2*L*dv + 2*L*L floats)

    psi = z @ p2["K"] * p2["theta"] + t * omega_short  # read phase             (L,)
    q = np.exp(-1j * psi)                              # (L,)
    o = (p2["w"] * q) @ S2 / L                         # (dv,)
    out_short[t] = np.concatenate([o.real, o.imag])

# check 1: sliding sum == windowed recompute (what a "sliding window attention" would do)
t = T - 1
ref = sum(kernel(p2, omega_short, t, s, Z[t], H[s]) * (Z[s] @ p2["V"]) for s in range(t - L + 1, t + 1))
assert np.allclose(out_short[t], np.concatenate([ref.real, ref.imag])), "short head != window recompute"

# check 2: with theta = 0 and w = 1 the kernel over the window is EXACTLY a delta at lag 0
lags = np.array([kernel(p2, omega_short, t, t - n, Z[t], H[t - n]).real for n in range(L)])
print(f"SHORT head: state = ring buffer of {L} tokens; sliding sum == window recompute OK")
print(f"            kernel over lags 0..{L-1} at theta=0, w=1: {np.round(lags, 6)}  <- Dirichlet delta")

# check 3: a learned w turns the same head into a causal FIR filter of length L (a short conv)
taps = rng.normal(size=L)                            # the filter you want over lags 0..L-1          (L,)
p2["w"] = (taps[:, None] * np.exp(1j * np.arange(L)[:, None] * omega_short[None])).sum(0)
                                                     # w_m = sum_n taps_n e^{+i n omega_m}           (L,)
lags = np.array([kernel(p2, omega_short, t, t - n, Z[t], H[t - n]).real for n in range(L)])
assert np.allclose(lags, taps), "kernel over lags != taps"
print(f"            with w_m = sum_n taps_n e^(i n omega_m): kernel over lags == taps -> o_t = sum_n taps_n e_(t-n), a {L}-tap conv;")
print(f"            theta != 0 makes the taps depend on content(z_t) - content(h_s): a selective window.")


# =============================================================================
# DAMPED LONG HEAD: rope grid, accumulator with PER-MODE DECAY, delta-rule write
# (a learned Laplace transform instead of a Fourier one)
# =============================================================================
# Same as the long head with one line changed: the state is damped before each
# write,  S <- diag(e^{-lambda}) S + c (x) e.  The codes keep |c_m| = 1, so
# ||c||^2 = M still holds and the delta rule's read-back needs no per-token
# normalisation.  Closed form:
#     o_t = sum_{s<=t} kappa_lam(t,s) e_s,
#     kappa_lam(t,s) = (1/M) sum_m w_m e^{-lambda_m (t-s)} e^{i(phi_s - psi_t)}
# so mode m forgets with time constant 1/lambda_m: slow modes keep, fast modes drop.
lam = np.exp(rng.uniform(np.log(0.01), np.log(0.5), size=M))   # decay per mode (M,), ~1/lam = 2..100 tokens
decay = np.exp(-lam)                                            # (M,)

def run_long(p, omega, lam, Z, H):
    """Token-by-token pass of the (damped) long head. lam = zeros -> the undamped head above."""
    M = len(omega)
    S = np.zeros((M, dv), complex); E = []; out = np.zeros((len(Z), 2 * dv))
    for t in range(len(Z)):
        z, h = Z[t], H[t]
        S = np.exp(-lam)[:, None] * S                        # DAMP the state first        (M, dv)
        c = np.exp(1j * (h @ p["K"] * p["theta"] + t * omega))   # write code               (M,)
        vhat = (np.conj(c) @ S).real / M                     # stored at this code (already damped)  (dv,)
        beta = sig(p["b"] @ z + p["b0"])
        e = z @ p["V"] - beta * vhat                         # error-corrected value       (dv,)
        S = S + np.outer(c, e); E.append(e)
        q = np.exp(-1j * (z @ p["K"] * p["theta"] + t * omega))
        o = (p["w"] * q) @ S / M
        out[t] = np.concatenate([o.real, o.imag])
    return out, E

def kernel_lam(p, omega, lam, t, s, zt, hs):
    """kappa with damping: each mode's contribution is scaled by e^{-lambda_m (t-s)}."""
    phi = hs @ p["K"] * p["theta"] + s * omega
    psi = zt @ p["K"] * p["theta"] + t * omega
    return (p["w"] * np.exp(-lam * (t - s)) * np.exp(1j * (phi - psi))).sum() / len(omega)

out_damp, E3 = run_long(p, omega_long, lam, Z, H)
t = T - 1
ref = sum(kernel_lam(p, omega_long, lam, t, s, Z[t], H[s]) * E3[s] for s in range(t + 1))
assert np.allclose(out_damp[t], np.concatenate([ref.real, ref.imag])), "damped head != closed form"
print(f"\nDAMPED head: same state {S.shape}, decay per mode 1/lambda in [{1/lam.max():.0f}, {1/lam.min():.0f}] tokens; closed form OK")

# --- what damping buys: a planted exact repeat, and how much of the read it gets
# Plant: the key at position t (= previous token H[t]) equals the key at s0 = t-30,
# with content phases on (theta != 0), and look at |kappa(t,s)|^2 over all s <= t.
# The trained model has |theta| = 2..11, which makes the codes a hash of the content
# (see CATCHUP.md); the toy's theta = 0.3 does not, so scale it up for this demo.
# 16 modes cannot separate 300 items whatever one does (noise energy N/M = 19 vs a match of
# at most 1), so the demo uses M = 64 (the real head has 190) and decays whose memory
# 1/lambda spans 20..200 tokens, so a match at lag 30 is not itself wiped out.
M2 = 64
omega_demo = 10000.0 ** (-np.arange(M2) / M2)
p_hash = make_params(M2); p_hash["theta"] = 3.0 * rng.normal(size=M2)   # |theta| ~ 3: codes decorrelate (hash)
lam_demo = np.exp(rng.uniform(np.log(1 / 200), np.log(1 / 20), size=M2))
T2 = 300; Z2 = rng.normal(size=(T2, d)); H2 = np.vstack([np.zeros((1, d)), Z2[:-1]])
t, s0 = T2 - 1, T2 - 1 - 30
H2[s0] = Z2[t]        # the key WRITTEN at s0 (previous token there) equals the token QUERYING at t: a match
for name, lam_ in (("undamped (Fourier)", np.zeros(M2)), ("damped   (Laplace)", lam_demo)):
    k2 = np.array([abs(kernel_lam(p_hash, omega_demo, lam_, t, s, Z2[t], H2[s])) ** 2 for s in range(t + 1)])
    share = k2[s0] / k2.sum(); n_eff = k2.sum() ** 2 / (k2 ** 2).sum()
    print(f"  {name}: planted match at lag 30 gets {share:5.1%} of the read energy; "
          f"effective #writes read = {n_eff:6.1f}  (of {t+1})")
print("  -> damping shrinks the crowd the match has to be heard over.  The two mechanisms differ in kind:\n"
      "     delta  = correction by SIMILARITY to the current write code (each write also reshapes the\n"
      "              older traces along its own direction, H <- (I - beta u u^T) H + u v^T);\n"
      "     damping = forgetting by AGE and spectral mode, whether or not a similar key ever returns.")


# =============================================================================
# KEY-VERIFIED long head: the write stores a copy of its own KEY beside the value,
# the read gets both back, and how well the key read back matches the query gates
# the output.  (== cdelta_kv; combined with damping == arm B)
# =============================================================================
# Why: neither the query z_t (gated_read) nor the read's magnitude (raw) can tell
# "found" from "nothing to find" -- measured on the trained model, read norms are
# the same on new and repeated words.  The read itself has to carry the evidence.
#
#     stored value   e_t = [ V(z_t) ; Kv(h_t) ]           (dv + dk)   key copy = f(the WRITE key h_t)
#     read           o_t = (1/M) sum_m w_m e^{-i psi_t} S_m   (dv + dk) complex, as before
#     evidence       m_t = cos( Re(o_t)[dv:] , Kv(z_t) )     scalar in [-1, 1]
#     gate           g_t = sigmoid(a . m_t + b)
#     output         [Re o_t ; Im o_t][value part] . g_t     (2 dv)   (the real layer RMS-normalises first)
#
# A genuine match wrote Kv(h_s) with h_s ~ z_t, so the key that comes back agrees
# with Kv(z_t).  A read over a mixture of unrelated writes comes back with an
# arbitrary key direction and m_t ~ 0.  Nothing else changes: same codes, same
# Gram, same delta rule (on the extended value), same decode.
dk = 16                                                      # the real head uses 16 too
Kv = rng.normal(size=(d, dk)) / np.sqrt(d)                  # key-copy projection      (d, dk)
a_g, b_g = 4.0, 0.0                                          # gate slope / bias (learned in the real head)

def run_long_kv(p, omega, lam, Z, H):
    M = len(omega)
    S = np.zeros((M, dv + dk), complex); out = np.zeros((len(Z), 2 * dv)); ev = np.zeros(len(Z))
    for t in range(len(Z)):
        z, h = Z[t], H[t]
        S = np.exp(-lam)[:, None] * S                        # damp (lam = 0 -> plain cdelta_kv)  (M, dv+dk)
        c = np.exp(1j * (h @ p["K"] * p["theta"] + t * omega))                              # (M,)
        vhat = (np.conj(c) @ S).real / M                     # stored at this code, value AND key parts (dv+dk,)
        beta = sig(p["b"] @ z + p["b0"])
        e = np.concatenate([z @ p["V"], h @ Kv]) - beta * vhat   # [value ; key copy] minus correction (dv+dk,)
        S = S + np.outer(c, e)
        q = np.exp(-1j * (z @ p["K"] * p["theta"] + t * omega))
        o = (p["w"] * q) @ S / M                             # (dv+dk,) complex
        key_back = o.real[dv:]                               # the key the memory returns          (dk,)
        key_want = z @ Kv                                    # the key a true match would return    (dk,)
        m = key_back @ key_want / (np.linalg.norm(key_back) * np.linalg.norm(key_want) + 1e-9)
        g = sig(a_g * m + b_g)
        ev[t] = m
        out[t] = np.concatenate([o.real[:dv], o.imag[:dv]]) * g   # value part only leaves the head  (2dv,)
    return out, ev

# --- does the evidence separate a match from no match?  Same planted sequence as above
# (key written at s0 = t-30 equals the token querying at t), hash-like codes, M = 64.
p_kv = dict(p_hash, V=rng.normal(size=(d, dv)) / np.sqrt(d), b=rng.normal(size=d) / np.sqrt(d), b0=-2.0)
for name, lam_ in (("undamped", np.zeros(M2)), ("damped  ", lam_demo)):
    _, ev = run_long_kv(p_kv, omega_demo, lam_, Z2, H2)
    others = np.delete(ev[64:], t - 64)                      # positions with nothing planted (beyond warm-up)
    print(f"KEY-VERIFIED head, {name}: evidence m at the planted match = {ev[t]:+.2f};  "
          f"at the other positions mean {others.mean():+.2f} sd {others.std():.2f}  "
          f"-> gate {sig(a_g*ev[t]+b_g):.2f} vs {sig(a_g*others.mean()+b_g):.2f}")
print("  -> the key that comes back agrees most with the query where a matching key was written.  The other\n"
      "     positions are not at 0 in this toy: with d = 8 random tokens are fairly alike, and a content-addressed\n"
      "     read returns the keys of PARTIAL matches too (theta = 0 gives ~0 there) -- that is the mechanism, not a\n"
      "     bug; at d = 128 the accidental similarity is far smaller.  Trained (diag_gate.py, layer 1 of the\n"
      "     combo): g = 0.59 on repeated words, 0.35 on new ones.")


# =============================================================================
# MIXTURE OF LAPLACE KERNELS: R read weights, chosen per token      (PROPOSED)
# =============================================================================
# WHAT LIMITS THE LAYER TODAY is not the write and not the state -- it is that
# ONE complex w gives ONE temporal kernel, shared by every one of the dv output
# channels and by every token. Composing the write and the read gives
#
#     kappa(t,s) = (1/M) sum_m w_m e^{-lambda_m (t-s)} e^{i[theta_m (K h_s - K z_t)_m + (s-t) omega_m]}
#
# and the only thing that varies with the token is the PHASE, through K z_t.
# The SHAPE of the kernel over lag -- how far back the layer looks, and with
# what profile -- is fixed at training time. Measured on the trained stack, the
# 16 profiles of an 8-layer model span a rank of 2.82: the layers learn nearly
# the same shape.
#
# The proposal keeps the state, the write, the transform and the complex
# structure exactly as they are, and changes only WHICH CONTOUR the read
# inverts along: R weight vectors instead of one, combined by coefficients the
# token itself produces.
#
#     o_t = sum_r alpha_r(z_t) * [ (w^(r) * q_t) @ S / M ]
#
# Because the read is linear in w, this is identical to reading once with an
# effective weight  w_eff(z_t) = sum_r alpha_r(z_t) w^(r)  -- a token-chosen
# point in the R-dimensional span of the learned weights. The kernel becomes
#
#     kappa_t(t,s) = sum_r alpha_r(z_t) kappa_r(t,s)
#
# still a sum of damped exponentials on the SAME modes: a mixture of Laplace
# kernels, not a departure from the transform. The state is untouched, which is
# the whole point -- 2M*dv floats whatever R is.
#
# Cost, measured at the campaign's shapes on a GB10: the read GEMM is 1.01% of
# a layer's forward, so R=4 is +3.0% of compute, R*M extra weights plus a d->R
# projection (+0.3% of parameters), and ZERO extra state.
R = 4
p3 = dict(p, w=None)                                  # same K, V, theta, beta as the long head
W = rng.normal(size=(R, M)) + 1j * rng.normal(size=(R, M))   # R read weights      (R, M)
A = rng.normal(size=(d, R)) / np.sqrt(d)              # alpha projection            (d, R)

S3 = np.zeros((M, dv), complex)                       # SAME state as the long head (M, dv)
E3, ALPHA = [], []
out_mix = np.zeros((T, 2 * dv))

for t in range(T):
    z, h = Z[t], H[t]
    phi = h @ p3["K"] * p3["theta"] + t * omega_long  # write phase, unchanged      (M,)
    c = np.exp(1j * phi)
    v = z @ p3["V"]
    vhat = (np.conj(c) @ S3).real / M                 # delta rule, unchanged       (dv,)
    beta = sig(p3["b"] @ z + p3["b0"])
    e = v - beta * vhat
    S3 = S3 + np.outer(c, e)                          # write, unchanged            (M, dv)
    E3.append(e)

    # --- the ONLY change: R reads, mixed by weights the token chooses
    a = np.exp(A.T @ z - (A.T @ z).max())
    alpha = a / a.sum()                               # softmax over R              (R,)
    ALPHA.append(alpha)
    psi = z @ p3["K"] * p3["theta"] + t * omega_long
    q = np.exp(-1j * psi)                             # read code, unchanged        (M,)
    o = ((alpha @ W) * q) @ S3 / M                    # == sum_r alpha_r (W_r*q)@S3 (dv,)
    out_mix[t] = np.concatenate([o.real, o.imag])

# check 1: mixing the WEIGHTS equals mixing the R separate READS (linearity)
t = T - 1
z = Z[t]
q = np.exp(-1j * (z @ p3["K"] * p3["theta"] + t * omega_long))
per_r = np.stack([(W[r] * q) @ S3 / M for r in range(R)])          # (R, dv)
assert np.allclose(ALPHA[t] @ per_r, (ALPHA[t] @ W * q) @ S3 / M), "mixture != R separate reads"

# check 2: the closed form is the SAME sum over the past, with a per-token kernel
def kernel_w(w, t, s, zt, hs):
    phi = hs @ p3["K"] * p3["theta"] + s * omega_long
    psi = zt @ p3["K"] * p3["theta"] + t * omega_long
    return (w * np.exp(1j * (phi - psi))).sum() / M
ref = sum(kernel_w(ALPHA[t] @ W, t, s, Z[t], H[s]) * E3[s] for s in range(t + 1))
assert np.allclose(out_mix[t], np.concatenate([ref.real, ref.imag])), "mixture != closed form"

# check 3: what it buys, isolated. A single w ALREADY varies with the token --
# through the phase theta_m (K h_s - K z_t)_m -- so comparing at theta != 0
# shows nothing: both look token-dependent. Set theta = 0 and the phase term
# collapses to (s-t) omega, identical for every token; then a single w gives
# ONE fixed shape over lag and the only remaining source of variation is the
# mixture. Centred rank 0 against R-1 is the whole claim.
lags = np.arange(12)
theta_keep = p3["theta"].copy()
p3["theta"] = np.zeros(M)                             # phase now content-free
def shapes_for(wfn):
    return np.stack([[kernel_w(wfn(t), t, t - n, Z[t], H[t - n]).real for n in lags]
                     for t in range(20, T)])          # (tokens, lags)
mix_shapes = shapes_for(lambda t: ALPHA[t] @ W)
one_shapes = shapes_for(lambda t: W[0])
r_mix = np.linalg.matrix_rank(mix_shapes - mix_shapes.mean(0), tol=1e-8)
r_one = np.linalg.matrix_rank(one_shapes - one_shapes.mean(0), tol=1e-8)
p3["theta"] = theta_keep
assert r_one == 0, "a single w must give one fixed shape once the phase is content-free"
print(f"MIXTURE head: state {S3.shape} complex = {2*S3.size} floats -- IDENTICAL to the long head's")
print(f"              mixing weights == mixing reads OK; closed form OK")
print(f"              at theta=0, centred rank of the lag-kernels: single w {r_one}, "
      f"R={R} mixture {r_mix} (<= R-1 = {R-1})")
print("  -> the state, the write and the transform are untouched; what the token now chooses is WHICH")
print("     Laplace kernel to invert along. The shape over lag stops being a training-time constant.")
