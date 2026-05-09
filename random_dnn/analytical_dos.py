"""Analytical singular value density of heavy-tailed MLP Jacobians.

Implements the Cizeau-Bouchaud real-axis cavity approach derived in
fig/analytical_dos.md. The density at each s equals the probability density
of the stable self-energy evaluated at -s, obtained after solving a
four-parameter (p, q, ptilde, qtilde) fixed-point system.

solve_fp_batch runs the fixed-point iteration simultaneously over all s values
in one vectorised loop: at each step the (M, n_samples) sample blocks are
drawn via the Chambers-Mallows-Stuck method, which takes a single (n_samples,)
draw of uniform and exponential noise and broadcasts across the M different
(beta_k, D_k) parameter pairs.

PDF evaluation uses a vectorised Zolotarev quadrature (_stable_pdf_batch).
The key factorisation g(theta; x0) = h(theta) * u^gamma and c2 * u = K
(both independent of x0, confirmed numerically) reduces the type-2 chi-average
from n_chi serial QUADPACK calls to n_theta Nolan.g setup evaluations followed
by a single (n_theta, n_chi) numpy broadcast.  scipy is retained for the
scalar type-1 evaluation where pointwise accuracy matters.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import special, stats

sys.path.insert(0, str(Path(__file__).parent))
from RMT import q_star_MC, stable_dist_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def C_alpha(alpha):
    """C_alpha = Gamma(1+alpha) sin(pi alpha/2) / pi."""
    return float(special.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / np.pi)


def _stable_pdf(x, alpha_s, beta, D):
    """PDF of L(alpha_s, beta, 0, D) at scalar or array x.

    Convention: CF = exp(-D |k|^alpha_s (1 - i beta sgn(k) tan(pi alpha_s/2))).
    scipy scale c satisfies D = c^alpha_s.
    """
    c    = float(D ** (1.0 / alpha_s))
    beta = float(np.clip(beta, -1.0, 1.0))
    return stats.levy_stable.pdf(np.asarray(x, dtype=float),
                                  alpha_s, beta, loc=0.0, scale=c)


def _zolotarev_eval(x_pos, alpha_s, beta, n_gl):
    """Evaluate L(alpha_s, beta, 0, 1) PDF in S1 parameterisation at x_pos > 0.

    scipy evaluates pdf = c2 * integral_{-xi}^{pi/2} g(t)*exp(-g(t)) dt where
    g(t; x_S0) = h(t)*u^gamma, u = x_S0 - zeta = x_S1 = x_pos, gamma < 0.
    The integrand peaks at g = 1, i.e. h(t) = u^{-gamma}.  scipy uses bisect
    to locate the peak and passes it as a hint to QUADPACK.

    Here we change variables to s = g = h(t)*u^gamma so the integrand becomes
    s*exp(-s) * phi(h=s/u^gamma) / u^gamma, where phi(h) = dtheta/dh is the
    Jacobian.  In s-space the integrand always peaks at s=1 regardless of u,
    so Gauss-Laguerre quadrature (weight exp(-s)) handles all u values with the
    same nodes.  phi(h) is precomputed once on a dense theta grid.
    """
    from scipy.stats._levy_stable import Nolan
    from scipy.special import roots_laguerre

    zeta  = -beta * np.tan(np.pi * alpha_s / 2)
    gamma = alpha_s / (alpha_s - 1)          # < 0 for alpha_s in (0, 1)
    u     = np.asarray(x_pos, dtype=float)   # all > 0

    nolan  = Nolan(alpha_s, beta, zeta + 1.0)
    xi, c2 = nolan.xi, nolan.c2

    # Build dense h(theta) table and compute phi(h) = dtheta/dh via finite differences.
    n_dense  = 4000
    eps      = 1e-8
    th_dense = np.linspace(-xi + eps, np.pi / 2 - eps, n_dense)
    h_dense  = np.array([nolan.g(t) for t in th_dense])   # monotone increasing

    dh       = np.diff(h_dense)
    dth      = th_dense[1] - th_dense[0]
    phi_mid  = dth / dh                                    # dtheta/dh > 0, (n_dense-1,)
    h_mid    = np.sqrt(h_dense[1:] * h_dense[:-1])        # geometric midpoints

    log_h_tbl  = np.log(h_mid)
    log_phi_tbl = np.log(phi_mid)

    # Gauss-Laguerre nodes t_k and weights w_k for integral_{0}^{inf} f(t) e^{-t} dt.
    t_gl, w_gl = roots_laguerre(n_gl)   # (n_gl,)

    # For each u: integral = (1/u^gamma) * sum_k w_k * t_k * phi(t_k / u^gamma)
    u_gamma = u ** gamma                              # (n_u,), < 1 for small u since gamma<0
    h_query = t_gl[:, None] / u_gamma[None, :]       # (n_gl, n_u), h values to look up

    # Log-space interpolation; clamp to table range (tails contribute ~ 0).
    log_h_q = np.log(np.maximum(h_query, h_mid[0]))
    log_h_q = np.minimum(log_h_q, log_h_tbl[-1])
    log_phi_q = np.interp(log_h_q.ravel(), log_h_tbl, log_phi_tbl).reshape(log_h_q.shape)
    phi_q = np.exp(log_phi_q)

    # Zero out contributions where h_query is outside the table (sub-machine-epsilon).
    phi_q[h_query < h_mid[0]]  = 0.0
    phi_q[h_query > h_mid[-1]] = 0.0

    intg = (w_gl[:, None] * t_gl[:, None] * phi_q).sum(axis=0)   # (n_u,)
    return c2 / u_gamma * intg


def _stable_pdf_batch(x_array, alpha_s, beta, D, n_theta=500):
    """Vectorised PDF of L(alpha_s, beta, 0, D) in S1 parameterisation.

    Evaluates at all x_array values simultaneously using the Zolotarev
    factorisation.  x > 0 (in D=1 standardised S1 space) uses _zolotarev_eval;
    x < 0 uses the S1 reflection p(x; beta) = p(-x; -beta); |x| < 0.01 uses
    the analytical value at x = 0 (x_S0 = zeta).
    """
    c    = float(D ** (1.0 / alpha_s))
    beta = float(np.clip(beta, -1.0, 1.0))
    xs   = np.asarray(x_array, dtype=float) / c   # S1 standardised (D = 1)

    # In S1 space: u = x_S1 = xs; boundary at u = 0 (x_S0 = zeta)
    zeta = -beta * np.tan(np.pi * alpha_s / 2)
    xi   = np.arctan(beta * np.tan(np.pi * alpha_s / 2)) / alpha_s

    result  = np.zeros_like(xs)
    tol     = 0.01
    at_zero = np.abs(xs) < tol
    above   = xs >=  tol
    below   = xs <= -tol

    if at_zero.any():
        # Analytical value at x_S0 = zeta (x_S1 = 0)
        result[at_zero] = (
            special.gamma(1 + 1 / alpha_s) * np.cos(xi)
            / np.pi / (1 + zeta ** 2) ** (1 / (2 * alpha_s))
        )

    if above.any():
        result[above] = _zolotarev_eval(xs[above], alpha_s, beta, n_theta)

    if below.any():
        # S1 reflection: p(x; beta) = p(-x; -beta); -xs[below] > 0
        result[below] = _zolotarev_eval(-xs[below], alpha_s, -beta, n_theta)

    return result / c


def _sample_stable(alpha_s, beta, D, size):
    """Sample from L(alpha_s, beta, 0, D) using torchlevy."""
    c    = float(D ** (1.0 / alpha_s))
    beta = float(np.clip(beta, -1.0, 1.0))
    return stable_dist_sample(alpha_s, beta, scale=c, size=size)


def _cms_sample_batch(alpha_s, beta_vec, D_vec, n_samples):
    """Vectorised CMS sampling from L(alpha_s, beta_k, 0, D_k) for k = 0..M-1.

    Uses the Chambers-Mallows-Stuck (1976) method. A single draw of
    (U, E) noise of shape (n_samples,) is shared across all M parameter
    pairs and broadcast to (M, n_samples) via the CMS formula.

    Valid for alpha_s in (0, 1) and (1, 2); here alpha_s = alpha/2 in (0.5, 1).

    Parameters
    ----------
    alpha_s : float
    beta_vec : (M,) tensor, skewness parameters in [-1, 1]
    D_vec    : (M,) tensor, exponent coefficients (> 0)
    n_samples : int

    Returns
    -------
    (M, n_samples) tensor
    """
    a   = float(alpha_s)
    eps = 1e-30

    U = torch.empty(n_samples).uniform_(-torch.pi / 2, torch.pi / 2)  # (n,)
    E = torch.empty(n_samples).exponential_()                          # (n,)

    tan_pa2 = float(np.tan(a * np.pi / 2))

    # CMS shape/location parameters, shape (M,)
    theta0 = torch.atan(beta_vec * tan_pa2) / a
    S      = (1.0 + (beta_vec * tan_pa2) ** 2) ** (1.0 / (2.0 * a))

    # Broadcast to (M, n_samples)
    U_  = U[None, :]        # (1, n)
    E_  = E[None, :]
    th_ = theta0[:, None]   # (M, 1)
    S_  = S[:, None]

    arg1 = a * (U_ + th_)
    arg2 = U_ - arg1

    # Standard CMS formula; cos(U) > 0 for U in (-pi/2, pi/2)
    X = (S_ * torch.sin(arg1)
         / torch.cos(U_).clamp(min=eps) ** (1.0 / a)
         * (torch.cos(arg2).abs().clamp(min=eps) / E_) ** ((1.0 - a) / a))

    # Scale from D=1 to D_k
    c = D_vec[:, None] ** (1.0 / a)   # (M, 1)
    return c * X                       # (M, n_samples)


# ---------------------------------------------------------------------------
# Chi samples (stationary derivative field)
# ---------------------------------------------------------------------------

def get_chi_samples(alpha, sigma_W, phi=torch.tanh, phi_prime=None,
                    n=int(1e5), seed=None):
    """Sample chi = phi'(h*) from the stationary-field distribution.

    h* ~ L(alpha, 0, 0, q*/2) at the MFT fixed point q*.
    phi_prime defaults to torch autograd of phi if not supplied.
    """
    if seed is not None:
        torch.manual_seed(seed)
    q_star = q_star_MC(alpha, sigma_W, phi=phi)[-1]
    if phi_prime is None:
        phi_prime = torch.vmap(torch.func.grad(phi))
    xi     = stable_dist_sample(alpha, scale=2.0 ** (-1.0 / alpha), size=n)
    h_star = float(q_star) ** (1.0 / alpha) * xi
    return phi_prime(h_star)   # (n,)


# ---------------------------------------------------------------------------
# Vectorised fixed-point solver (all s values simultaneously)
# ---------------------------------------------------------------------------

def solve_fp_batch(s_vals, alpha, sigma_W, chi,
                   n_samples=int(1e4), n_iter=200, tol=1e-4):
    """Vectorised fixed-point iteration over all s values simultaneously.

    At each step a single (n_samples,) noise draw is shared across all M
    s values via _cms_sample_batch, giving (M, n_samples) sample blocks.
    All moment computations are then batched over (M, n_samples) tensors.
    Iteration stops when every s value has converged.

    Parameters
    ----------
    s_vals   : array-like of shape (M,)
    alpha, sigma_W : float
    chi      : (n_chi,) torch tensor of phi'(h*) samples
    n_samples : MC samples per iteration step
    n_iter, tol : convergence controls

    Returns
    -------
    p, q, ptilde, qtilde : four (M,) tensors
    """
    alpha_s = alpha / 2.0
    Calpha  = C_alpha(alpha)
    s_t     = torch.as_tensor(np.asarray(s_vals, dtype=np.float32))  # (M,)

    # Initialise from large-s asymptotics: G ~ -1/s so p ~ |s|^{-alpha_s}
    s_abs  = s_t.abs().clamp(min=1e-4)
    chi_am = float((chi.abs() ** alpha).mean())
    p      = s_abs ** (-alpha_s)
    q      = chi_am * s_abs ** (-alpha_s)
    ptilde = -p.clone()
    qtilde = -q.clone()

    # Fixed chi subsample, shared across all s and all iterations
    idx     = torch.randint(0, len(chi), (n_samples,))
    chi_sub = chi[idx]                       # (n,)
    chi_sq  = chi_sub ** 2                   # (n,)
    chi_al  = chi_sub.abs() ** (2 * alpha_s) # |chi|^alpha, (n,)
    eps     = 1e-30

    for _ in range(n_iter):
        D_A    = (Calpha * sigma_W ** alpha / 2.0) * q.clamp(min=eps)   # (M,)
        D_C    = (Calpha * sigma_W ** alpha / 2.0) * p.clamp(min=eps)   # (M,)
        beta_A = (qtilde / q.clamp(min=1e-12)).clamp(-1.0, 1.0)         # (M,)
        beta_C = (ptilde / p.clamp(min=1e-12)).clamp(-1.0, 1.0)         # (M,)

        # One CMS draw gives (M, n_samples) sample blocks
        A = _cms_sample_batch(alpha_s, beta_A, D_A, n_samples)  # (M, n)
        C = _cms_sample_batch(alpha_s, beta_C, D_C, n_samples)  # (M, n)

        # Type-1 moments: vectorised over (M, n_samples)
        sA      = s_t[:, None] + A
        abs_sA  = sA.abs().clamp(min=eps)
        p_new      = (abs_sA  ** (-alpha_s)).mean(dim=1)                         # (M,)
        ptilde_new = -(sA.sign() * abs_sA ** (-alpha_s)).mean(dim=1)             # (M,)

        # Type-2 moments: vectorised over (M, n_samples)
        schiC      = s_t[:, None] + chi_sq[None, :] * C
        abs_schiC  = schiC.abs().clamp(min=eps)
        q_new      = (chi_al[None, :] * abs_schiC ** (-alpha_s)).mean(dim=1)             # (M,)
        qtilde_new = -(chi_al[None, :] * schiC.sign() * abs_schiC ** (-alpha_s)).mean(dim=1)

        converged = (
            ((p_new - p).abs() < tol * (p.abs() + 1e-10)) &
            ((q_new - q).abs() < tol * (q.abs() + 1e-10))
        ).all()
        p, q, ptilde, qtilde = p_new, q_new, ptilde_new, qtilde_new
        if converged:
            break

    return p, q, ptilde, qtilde


# ---------------------------------------------------------------------------
# Single-s solver (warm-startable, useful for debugging)
# ---------------------------------------------------------------------------

def solve_fp(s, alpha, sigma_W, chi, n_samples=int(1e4), n_iter=200, tol=1e-4,
             p0=None, q0=None, ptilde0=None, qtilde0=None):
    """Fixed-point iteration for a single s value. Supports warm-starting."""
    alpha_s = alpha / 2.0
    Calpha  = C_alpha(alpha)
    s       = float(s)
    s_abs   = max(abs(s), 1e-4)
    chi_am  = float((chi.abs() ** alpha).mean())

    p      = p0      if p0      is not None else s_abs ** (-alpha_s)
    q      = q0      if q0      is not None else chi_am * s_abs ** (-alpha_s)
    ptilde = ptilde0 if ptilde0 is not None else -p
    qtilde = qtilde0 if qtilde0 is not None else -q

    idx     = torch.randint(0, len(chi), (n_samples,))
    chi_sub = chi[idx]

    eps = 1e-30
    for _ in range(n_iter):
        D_A    = Calpha * sigma_W ** alpha * max(q, 1e-30) / 2.0
        D_C    = Calpha * sigma_W ** alpha * max(p, 1e-30) / 2.0
        beta_A = float(np.clip(qtilde / q if abs(q) > 1e-12 else 0.0, -1.0, 1.0))
        beta_C = float(np.clip(ptilde / p if abs(p) > 1e-12 else 0.0, -1.0, 1.0))

        A = _sample_stable(alpha_s, beta_A, D_A, n_samples)
        C = _sample_stable(alpha_s, beta_C, D_C, n_samples)

        sA      = s + A;  abs_sA = sA.abs().clamp(min=eps)
        p_new      = (abs_sA ** (-alpha_s)).mean().item()
        ptilde_new = -(sA.sign() * abs_sA ** (-alpha_s)).mean().item()

        chi_sq  = chi_sub ** 2
        schiC   = s + chi_sq * C;  abs_sCC = schiC.abs().clamp(min=eps)
        chi_al  = chi_sub.abs() ** (2 * alpha_s)
        q_new      = (chi_al * abs_sCC ** (-alpha_s)).mean().item()
        qtilde_new = -(chi_al * schiC.sign() * abs_sCC ** (-alpha_s)).mean().item()

        converged = (abs(p_new - p) < tol * (abs(p) + 1e-10) and
                     abs(q_new - q) < tol * (abs(q) + 1e-10))
        p, q, ptilde, qtilde = p_new, q_new, ptilde_new, qtilde_new
        if converged:
            break

    return p, q, ptilde, qtilde


# ---------------------------------------------------------------------------
# DOS evaluation
# ---------------------------------------------------------------------------

def dos_point(s, alpha, sigma_W, chi, p, q, ptilde, qtilde):
    """Eigenvalue density of H at s given converged FP parameters.

    rho_H(s) = 0.5 * (p_A(-s) + <p_{chi^2 C}(-s)>_chi)

    The singular-value density of A is 2 * rho_H(s) for s > 0; dos() applies
    this factor so callers of dos() receive the singular-value density directly.
    """
    alpha_s = alpha / 2.0
    Calpha  = C_alpha(alpha)
    s       = float(s)
    p, q, ptilde, qtilde = float(p), float(q), float(ptilde), float(qtilde)

    D_A    = Calpha * sigma_W ** alpha * max(q, 1e-30) / 2.0
    D_C    = Calpha * sigma_W ** alpha * max(p, 1e-30) / 2.0
    beta_A = float(np.clip(qtilde / q if abs(q) > 1e-12 else 0.0, -1.0, 1.0))
    beta_C = float(np.clip(ptilde / p if abs(p) > 1e-12 else 0.0, -1.0, 1.0))

    rho1 = float(_stable_pdf(-s, alpha_s, beta_A, D_A))

    chi_np = chi.cpu().numpy()
    chi_sq = chi_np ** 2
    mask   = chi_sq > 1e-12
    rho2   = (float(np.mean(_stable_pdf(-s / chi_sq[mask], alpha_s, beta_C, D_C)
                             / chi_sq[mask]))
              if mask.any() else 0.0)

    return 0.5 * (rho1 + rho2)


def dos(s_vals, alpha, sigma_W, chi, n_samples=int(1e4), n_iter=200, tol=1e-4,
        n_theta=500, verbose=False):
    """Analytical singular-value density at each s in s_vals.

    Runs solve_fp_batch (vectorised over all s simultaneously), then evaluates
    the stable PDF at each grid point.  Type-1 (scalar) uses scipy; type-2
    (chi-average over n_chi points) uses _stable_pdf_batch with n_theta
    Zolotarev quadrature nodes.

    Returns ndarray of shape (len(s_vals),) — the singular-value density
    (= 2 * eigenvalue density of the Hermitian bipartisation H).
    """
    s_vals = np.asarray(s_vals, dtype=float)

    if verbose:
        print(f"Vectorised FP over {len(s_vals)} s-values ...", flush=True)
    p_v, q_v, pt_v, qt_v = solve_fp_batch(
        s_vals, alpha, sigma_W, chi,
        n_samples=n_samples, n_iter=n_iter, tol=tol,
    )

    # Precompute chi quantities once (avoids repeated .cpu().numpy() and masking).
    alpha_s  = alpha / 2.0
    Calpha   = C_alpha(alpha)
    chi_np   = chi.cpu().numpy()
    chi_sq   = chi_np ** 2
    mask     = chi_sq > 1e-12
    chi_sq_m = chi_sq[mask]

    if verbose:
        print("Evaluating PDF at grid points ...", flush=True)
    rho = np.empty(len(s_vals))
    for k, (s, p, q, pt, qt) in enumerate(zip(s_vals,
                                               p_v.tolist(), q_v.tolist(),
                                               pt_v.tolist(), qt_v.tolist())):
        D_A    = Calpha * sigma_W ** alpha * max(q,  1e-30) / 2.0
        D_C    = Calpha * sigma_W ** alpha * max(p,  1e-30) / 2.0
        beta_A = float(np.clip(qt / q  if abs(q) > 1e-12 else 0.0, -1.0, 1.0))
        beta_C = float(np.clip(pt / p  if abs(p) > 1e-12 else 0.0, -1.0, 1.0))

        rho1 = float(_stable_pdf(-s, alpha_s, beta_A, D_A))
        rho2 = (float(np.mean(
                    _stable_pdf_batch(-s / chi_sq_m, alpha_s, beta_C, D_C, n_theta)
                    / chi_sq_m))
                if mask.any() else 0.0)
        rho[k] = 0.5 * (rho1 + rho2)

    return 2.0 * rho   # singular-value density = 2 * H eigenvalue density


# ---------------------------------------------------------------------------
# CLI (same keyword-arg convention as RMT.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from time import time
    tic     = time()
    func    = eval(sys.argv[1])
    args    = [sys.argv[i: i + 3] for i in range(2, len(sys.argv), 3)]
    arg_dict = {a[0]: eval(a[2])(a[1]) for a in args}
    result  = func(**arg_dict)
    if result is not None:
        print(result)
    print(f"Script time: {time() - tic:.2f} sec")
