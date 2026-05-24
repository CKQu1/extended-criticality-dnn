"""Layerwise Jacobian singular-value distribution for heavy-tailed MLPs.

Implements pathways (P1) Theorem-2 deterministic-quantile theory,
(P2) population-dynamics cavity-equation route (wraps RMT.jac_cavity_svd_log_pdf),
(P3a) synthetic empirical SVDs (h_j drawn directly from S* p_alpha),
(P3b) MLP-derived empirical SVDs (wraps RMT.MLP, last-layer postjac).
See ht_mlp_jacobian.md for the derivation.

CLI:
    python ht_mlp_jacobian.py run_validation alpha 1.5 float sigma_W 1.0 float \\
                              N 256 int num_matrices 80 int depth 60 int seed 0 int
    python ht_mlp_jacobian.py convention_check alpha 1.5 float sigma_W 1.0 float \\
                              N 256 int num_matrices 80 int seed 0 int
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Callable, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "RMT"))
sys.path.insert(0, str(_HERE))

import structured_wishart_levy as swl  # noqa: E402  -- empirical sampler only
import one_sided_wishart_levy as osw  # noqa: E402  -- Theorem 2 scalar solver
import wishart_levy as wl  # noqa: E402
import torch  # noqa: E402

import RMT  # noqa: E402  -- random_dnn/RMT.py (top-level), not the RMT/ dir


# ---------------------------------------------------------------------------
# Timer (per CLAUDE.md: time subparts so bottlenecks are visible at a glance).
# ---------------------------------------------------------------------------


@contextmanager
def Timer(label: str, log: Optional[list] = None) -> None:
    tic = time()
    yield
    dt = time() - tic
    msg = f"[{label}] elapsed {dt:.3f}s"
    print(msg)
    if log is not None:
        log.append((label, dt))


# ---------------------------------------------------------------------------
# Quantile of |phi'(S* Z)|, Z ~ p_alpha (Belinschi).
# ---------------------------------------------------------------------------


def _belinschi_stable_samples(
    alpha: float, n_samples: int, seed: Optional[int] = None
) -> np.ndarray:
    """Draw n_samples from p_alpha (Belinschi: char fn exp(-|t|^a / 2)).

    SciPy's stats.levy_stable with scale=1 has char fn exp(-|t|^a).  To get the
    Belinschi convention we rescale by 2**(-1/alpha) (matches the convention
    used in RMT.py:MFT_map / q_star_MC).
    """
    from scipy import stats as _stats

    rng = np.random.default_rng(seed)
    return np.asarray(
        _stats.levy_stable.rvs(alpha, 0.0, loc=0.0, scale=2.0 ** (-1.0 / alpha),
                               size=n_samples, random_state=rng),
        dtype=float,
    )


@dataclass
class JacobianProfile:
    alpha: float
    sigma_w: float
    q_star: float
    S_star: float
    profile_samples: np.ndarray   # sorted ascending |phi'(S* Z)| samples
    profile_alpha_moment: float   # E|phi'(S* Z)|^a


def jacobian_profile(
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    q_star: Optional[float] = None,
    n_samples: int = 50_000,
    seed: Optional[int] = None,
) -> JacobianProfile:
    """Build the deterministic column profile of section 2 / Lemma 1.

    Returns sorted samples of |phi'(S* Z)|, Z ~ p_alpha, which serve as an
    empirical quantile function (see `quantile_callable`).
    """
    if q_star is None:
        q_star = RMT.q_star_MC(alpha, sigma_w, sigma_b=sigma_b, phi=phi,
                               seed=seed)[-1]
    S_star = q_star ** (1.0 / alpha)
    z = _belinschi_stable_samples(alpha, n_samples, seed=seed)
    phi_prime = torch.func.vmap(torch.func.grad(phi))
    samples = phi_prime(torch.tensor(S_star * z)).abs().cpu().numpy()
    samples = np.sort(samples)
    moment = float(np.mean(samples ** alpha))
    return JacobianProfile(
        alpha=float(alpha), sigma_w=float(sigma_w), q_star=float(q_star),
        S_star=float(S_star), profile_samples=samples,
        profile_alpha_moment=moment,
    )


def quantile_callable(profile: JacobianProfile) -> Callable[[np.ndarray], np.ndarray]:
    """Empirical quantile c(v) = F^{-1}(v) from sorted samples.

    Returns a callable accepting a 1-D or 2-D array of v in [0, 1].
    """
    samples = profile.profile_samples
    n = samples.size
    # Cell-centre convention: sample k (0-indexed) sits at v = (k + 0.5) / n.
    grid = (np.arange(n) + 0.5) / n

    def c(v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, dtype=float), 0.0, 1.0)
        return np.interp(v, grid, samples)

    return c


# ---------------------------------------------------------------------------
# (P1) Theorem-2 deterministic-quantile theory.
# ---------------------------------------------------------------------------


def theoretical_jacobian_sv_curve(
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    s_max: float = 8.0,
    num_points: int = 161,
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    profile_order: int = 64,
    profile: Optional[JacobianProfile] = None,
    n_profile_samples: int = 50_000,
    seed: Optional[int] = None,
):
    """(P1) Theoretical SV density of the layerwise Jacobian J^l = D^l W^l.

    Wraps `one_sided_wishart_levy.theoretical_one_sided_singular_value_curve`
    with gamma=1, entry_scale=sigma_w, and c(y) the deterministic quantile of
    |phi'(S* Z)|.  Theorem 2 scalar closure with independent-rule h_alpha
    readout (see RMT/one_sided_wishart_levy.md).  Restricted to alpha < 2.
    """
    if not (1.0 < alpha < 2.0):
        raise ValueError("alpha in (1, 2) only (heavy-tailed branch).  "
                         "For alpha=2 use the Gaussian / Marchenko-Pastur "
                         "branch separately (md sec. 4).")
    if profile is None:
        profile = jacobian_profile(alpha, sigma_w, phi=phi, sigma_b=sigma_b,
                                   n_samples=n_profile_samples, seed=seed)
    c = quantile_callable(profile)
    curve = osw.theoretical_one_sided_singular_value_curve(
        alpha=alpha, gamma=1.0, c=c,
        s_max=s_max, num_points=num_points,
        entry_scale=sigma_w, normalization="stable",
        imag_eps=imag_eps, quadrature_order=quadrature_order,
        profile_order=profile_order,
    )
    return curve, profile


def theoretical_tail_constant(profile: JacobianProfile) -> float:
    """Tail constant B in f_SV(s) ~ B s^{-1-alpha}, eq. (7) of the md.

    B = (alpha/2) * sigma_w^alpha * E|phi'(S* Z)|^alpha * scale^alpha,
    where scale = belinschi_quantile_scale(alpha, entry_scale=sigma_w) folds in
    the SciPy <-> Belinschi tail-amplitude conversion used by the structured
    curve.  Atomless convention (gamma=1, no atom).
    """
    alpha = profile.alpha
    scale = wl.belinschi_quantile_scale(alpha, entry_scale=profile.sigma_w)
    return float(0.5 * alpha * profile.profile_alpha_moment * scale ** alpha)


# ---------------------------------------------------------------------------
# (P2) Population-dynamics cavity-equation route (wraps RMT.py).
# ---------------------------------------------------------------------------


def population_dynamics_sv_density(
    sing_vals: np.ndarray,
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    q_star: Optional[float] = None,
    num_doublings: int = 8,
    num_chis: int = 1,
    progress: bool = False,
    seed: Optional[int] = None,
    num_steps_fn: Optional[Callable[[int], int]] = None,
):
    """(P2) Linear-space SV density from RMT.jac_cavity_svd_log_pdf.

    The underlying function returns log|density|; we exponentiate and
    aggregate over the num_chis realisations (mean + std).  Compare in
    linear space, not log (the advisor flag: log amplifies near-zero noise).

    ``num_steps_fn`` controls cavity iterations per doubling stage; default
    is RMT.py's ``lambda P: P**2`` (iterations-per-element ~ P, fast
    cavity mixing).  For the pool-size scaling diagnostic that needs to
    decouple mixing from pool-mean MC noise, pass a fn like
    ``lambda P: c * P`` so each element is updated a constant ~c times
    regardless of P.
    """
    kwargs = dict(
        sigma_b=sigma_b, phi=phi, q=q_star,
        num_doublings=num_doublings, num_chis=num_chis,
        progress=progress, seed=seed,
    )
    if num_steps_fn is not None:
        kwargs["num_steps_fn"] = num_steps_fn
    log_pdf = RMT.jac_cavity_svd_log_pdf(
        np.asarray(sing_vals, dtype=float), alpha=alpha, sigma_W=sigma_w,
        **kwargs,
    )
    pdf = np.exp(log_pdf)
    if pdf.ndim == 1:
        return pdf, np.zeros_like(pdf)
    return pdf.mean(axis=1), pdf.std(axis=1)


# ---------------------------------------------------------------------------
# (P3a) Synthetic empirical SVDs: h_j drawn directly from S* p_alpha.
# ---------------------------------------------------------------------------


def _scipy_stable_matrix(
    alpha: float, shape: tuple[int, int], seed: Optional[int] = None
) -> np.ndarray:
    """Draw a matrix of i.i.d. SciPy unit-scale symmetric alpha-stable entries.

    SciPy convention (char fn exp(-|t|^a)) matches what `RMT.py:MLP` actually
    samples (torchlevy default = SciPy convention, since `q_star_MC` /
    `MFT_map` explicitly pass `scale=2**(-1/alpha)` to *convert* to
    Belinschi).  The Belinschi-vs-SciPy factor of 2**(-1/alpha) is absorbed
    into the matrix prefactor (sigma_w * (2N)**(-1/alpha) = sigma_w *
    N**(-1/alpha) * 2**(-1/alpha)) by the MLP code.
    """
    from scipy import stats as _stats

    rng = np.random.default_rng(seed)
    return np.asarray(
        _stats.levy_stable.rvs(alpha, 0.0, loc=0.0, scale=1.0,
                               size=shape, random_state=rng),
        dtype=float,
    )


def synthetic_jacobian_sv_spectrum(
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    q_star: Optional[float] = None,
    N: int = 400,
    num_matrices: int = 60,
    seed: Optional[int] = None,
    bins: int = 121,
    sv_range: Optional[tuple[float, float]] = None,
):
    """(P3a) Synthetic empirical SVs of J = D W with h_j ~ S* p_alpha.

    W has entries sigma_w * (2N)^{-1/alpha} * Belinschi-stable -- matches the
    RMT.py:MLP convention (line 87-91 of RMT.py).  Returns histogram
    centres and density.
    """
    if q_star is None:
        q_star = RMT.q_star_MC(alpha, sigma_w, sigma_b=sigma_b, phi=phi,
                               seed=seed)[-1]
    S_star = q_star ** (1.0 / alpha)
    rng = np.random.default_rng(seed)
    phi_prime = torch.func.vmap(torch.func.grad(phi))
    w_scale = sigma_w * (2.0 * N) ** (-1.0 / alpha)
    all_sv = []
    for _ in range(num_matrices):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        h = S_star * _belinschi_stable_samples(alpha, N, seed=sub_seed)
        d = phi_prime(torch.tensor(h)).cpu().numpy()
        W = w_scale * _scipy_stable_matrix(alpha, (N, N), seed=sub_seed + 1)
        J = d[:, None] * W                      # D W
        s = np.linalg.svd(J, compute_uv=False)
        all_sv.append(s)
    sv = np.concatenate(all_sv)
    density, edges = np.histogram(sv, bins=bins, range=sv_range, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return {
        "sv": sv,
        "centres": centres,
        "density": density,
        "edges": edges,
        "q_star": float(q_star),
        "S_star": float(S_star),
        "N": int(N),
        "num_matrices": int(num_matrices),
    }


# ---------------------------------------------------------------------------
# (P3b) MLP-derived empirical SVDs: wraps RMT.py:MLP, reads last-layer postjac.
# ---------------------------------------------------------------------------


def mlp_jacobian_sv_spectrum(
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    N: int = 256,
    depth: int = 60,
    num_matrices: int = 40,
    burn_in: int = 20,
    seed: Optional[int] = None,
    bins: int = 121,
    sv_range: Optional[tuple[float, float]] = None,
    q_star: Optional[float] = None,
):
    """(P3b) Empirical SVs of postjac from RMT.MLP across realisations.

    For each realisation runs an MLP of given depth with a random input,
    collects postjac singular values at layers in [burn_in, depth) (where
    q^l has converged to q^*), pools across realisations.  Also returns
    the empirical |q^L - q^*|/q^* ratio as a forward-convergence self-test.
    """
    if q_star is None:
        q_star = RMT.q_star_MC(alpha, sigma_w, sigma_b=sigma_b, phi=phi,
                               seed=seed)[-1]
    rng = np.random.default_rng(seed)
    all_sv = []
    q_l_samples = []
    for _ in range(num_matrices):
        x0 = rng.standard_normal(N).astype(np.float32)
        sub_seed = int(rng.integers(0, 2**31 - 1))
        stats = RMT.MLP(x0, depth=depth, alpha=alpha, sigma_W=sigma_w,
                        sigma_b=sigma_b, phi=phi, seed=sub_seed,
                        fast=False, usetqdm=False)
        # last layers (post burn-in)
        log_sv = stats["postjac_log_svdvals"][burn_in:]    # (depth - burn_in, N)
        sv = np.exp(log_sv).reshape(-1)
        all_sv.append(sv)
        # forward q-convergence check: q^l = E|phi(h^l)|^alpha at last layer
        postact = stats["postact"][-1]
        q_l_samples.append(
            sigma_w ** alpha * float(np.mean(np.abs(postact) ** alpha))
            + sigma_b ** alpha
        )
    sv = np.concatenate(all_sv)
    density, edges = np.histogram(sv, bins=bins, range=sv_range, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    q_L_mean = float(np.mean(q_l_samples))
    q_rel_err = float(abs(q_L_mean - q_star) / max(q_star, 1e-12))
    return {
        "sv": sv,
        "centres": centres,
        "density": density,
        "edges": edges,
        "q_star": float(q_star),
        "q_L_mean": q_L_mean,
        "q_rel_err": q_rel_err,
        "N": int(N),
        "depth": int(depth),
        "burn_in": int(burn_in),
        "num_matrices": int(num_matrices),
    }


# ---------------------------------------------------------------------------
# D_q (multifractal exponent) of left/right SVs.
#
# Two pathways:
#   (P3b-Dq) mlp_jacobian_Dq_spectrum -- empirical D_q from RMT.MLP forward
#            with compute_uv=True at post-burn-in layers.
#   (Pthy-Dq) synthetic_jacobian_Dq_spectrum -- "theory" curve from the
#            BDG-predicted matrix ensemble: sample h ~ S* p_alpha directly,
#            build J = D W, SVD.  No analytical closed form is available
#            for D_q from Y_r alone (RMT/localisation.md sec. 9); the
#            synthetic-ensemble curve is the cleanest matrix-level theory
#            comparison.  Agreement with (P3b-Dq) means the BDG matrix
#            ensemble correctly models the MLP Jacobian.
# ---------------------------------------------------------------------------


def mlp_jacobian_Dq_spectrum(
    alpha: float,
    sigma_w: float,
    q_grid: np.ndarray,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    N: int = 256,
    depth: int = 50,
    num_matrices: int = 10,
    burn_in: int = 25,
    seed: Optional[int] = None,
    sv_bins: int = 21,
    sv_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Empirical D_q^{left}(s), D_q^{right}(s) for the postjac J = D W.

    For each MLP realisation runs RMT.MLP with compute_uv=True, collects
    left and right singular vectors at layers in [burn_in, depth), computes
    per-vector D_q = -log(sum_i |v(i)|^{2q}) / ((q-1) log N), and bins by
    singular value.  Pool across realisations and post-burn-in layers.

    Phenomenology only.  Compares against future theoretical D_q curves
    (model-dependent; see RMT/localisation.md sec. 9F).
    """
    q_grid = np.asarray(q_grid, dtype=float)
    rng = np.random.default_rng(seed)
    log_N = float(np.log(N))

    # Pre-scan to set sv_range if not given
    if sv_range is None:
        x0 = rng.standard_normal(N).astype(np.float32)
        sub_seed = int(rng.integers(0, 2**31 - 1))
        stats0 = RMT.MLP(x0, depth=depth, alpha=alpha, sigma_W=sigma_w,
                         sigma_b=sigma_b, phi=phi, seed=sub_seed,
                         fast=False, compute_uv=False, usetqdm=False)
        log_sv0 = stats0["postjac_log_svdvals"][burn_in:]
        sv_all0 = np.exp(log_sv0).reshape(-1)
        sv_range = (0.0, float(np.percentile(sv_all0, 99.0)))

    sv_edges = np.linspace(sv_range[0], sv_range[1], int(sv_bins) + 1)
    sv_centres = 0.5 * (sv_edges[1:] + sv_edges[:-1])

    Dq_left_sum = np.zeros((int(sv_bins), q_grid.size), dtype=float)
    Dq_right_sum = np.zeros((int(sv_bins), q_grid.size), dtype=float)
    bin_count = np.zeros(int(sv_bins), dtype=int)

    for _ in range(num_matrices):
        x0 = rng.standard_normal(N).astype(np.float32)
        sub_seed = int(rng.integers(0, 2**31 - 1))
        stats = RMT.MLP(x0, depth=depth, alpha=alpha, sigma_W=sigma_w,
                        sigma_b=sigma_b, phi=phi, seed=sub_seed,
                        fast=False, compute_uv=True, usetqdm=False)
        log_sv = stats["postjac_log_svdvals"][burn_in:]      # (L', N)
        # svd_left / svd_right have rows-are-vectors layout (RMT.py line 110-111)
        u_layers = stats["postjac_svd_left"][burn_in:]        # (L', N, N)
        v_layers = stats["postjac_svd_right"][burn_in:]       # (L', N, N)

        for ell in range(log_sv.shape[0]):
            sv_vals = np.exp(log_sv[ell])                     # (N,)
            U_abs2 = np.abs(u_layers[ell]) ** 2               # (N, N), rows=vectors
            V_abs2 = np.abs(v_layers[ell]) ** 2               # (N, N), rows=vectors
            for qi, q in enumerate(q_grid):
                if abs(q - 1.0) < 1e-12:
                    Dq_left = np.ones(int(N))
                    Dq_right = np.ones(int(N))
                else:
                    I_q_left = (U_abs2 ** float(q)).sum(axis=-1)
                    I_q_right = (V_abs2 ** float(q)).sum(axis=-1)
                    I_q_left = np.where(I_q_left > 0, I_q_left, np.nan)
                    I_q_right = np.where(I_q_right > 0, I_q_right, np.nan)
                    Dq_left = -np.log(I_q_left) / ((q - 1.0) * log_N)
                    Dq_right = -np.log(I_q_right) / ((q - 1.0) * log_N)
                s_bin = np.digitize(sv_vals, sv_edges) - 1
                for k in range(int(N)):
                    sb = s_bin[k]
                    if 0 <= sb < int(sv_bins):
                        if np.isfinite(Dq_left[k]):
                            Dq_left_sum[sb, qi] += float(Dq_left[k])
                        if np.isfinite(Dq_right[k]):
                            Dq_right_sum[sb, qi] += float(Dq_right[k])
                        if qi == 0:
                            bin_count[sb] += 1

    safe = bin_count > 0
    Dq_left_mean = np.full_like(Dq_left_sum, np.nan)
    Dq_right_mean = np.full_like(Dq_right_sum, np.nan)
    Dq_left_mean[safe] = Dq_left_sum[safe] / bin_count[safe, None]
    Dq_right_mean[safe] = Dq_right_sum[safe] / bin_count[safe, None]

    return {
        "sv_centres": sv_centres,
        "sv_edges": sv_edges,
        "q_grid": q_grid,
        "Dq_left_mean": Dq_left_mean,
        "Dq_right_mean": Dq_right_mean,
        "bin_count": bin_count,
        "N": int(N),
        "depth": int(depth),
        "burn_in": int(burn_in),
        "num_matrices": int(num_matrices),
    }


def synthetic_jacobian_Dq_spectrum(
    alpha: float,
    sigma_w: float,
    q_grid: np.ndarray,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    q_star: Optional[float] = None,
    N: int = 256,
    num_matrices: int = 40,
    seed: Optional[int] = None,
    sv_bins: int = 21,
    sv_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Theoretical-ensemble D_q^{left}(s), D_q^{right}(s) for J = D W with
    h ~ S* p_alpha drawn directly (no MLP forward).

    This is the cleanest "theory" curve we can produce at the matrix-ensemble
    level: it isolates the BDG-predicted random matrix from MLP-forward
    structure.  If this matches `mlp_jacobian_Dq_spectrum`, the BDG ensemble
    is the right matrix model for the MLP Jacobian; deviations identify
    structure the BDG framework doesn't capture.

    No closed-form analytic prediction is provided (see
    RMT/localisation.md sec. 9F-G for the obstruction); the synthetic
    SVDs play the role of "theory" here.
    """
    q_grid = np.asarray(q_grid, dtype=float)
    if q_star is None:
        q_star = RMT.q_star_MC(alpha, sigma_w, sigma_b=sigma_b, phi=phi,
                               seed=seed)[-1]
    S_star = q_star ** (1.0 / alpha)
    rng = np.random.default_rng(seed)
    log_N = float(np.log(N))
    phi_prime = torch.func.vmap(torch.func.grad(phi))
    w_scale = sigma_w * (2.0 * N) ** (-1.0 / alpha)

    if sv_range is None:
        sub_seed0 = int(rng.integers(0, 2**31 - 1))
        h0 = S_star * _belinschi_stable_samples(alpha, N, seed=sub_seed0)
        d0 = phi_prime(torch.tensor(h0)).cpu().numpy()
        W0 = w_scale * _scipy_stable_matrix(alpha, (N, N), seed=sub_seed0 + 1)
        s0 = np.linalg.svd(d0[:, None] * W0, compute_uv=False)
        sv_range = (0.0, float(np.percentile(s0, 99.0)))

    sv_edges = np.linspace(sv_range[0], sv_range[1], int(sv_bins) + 1)
    sv_centres = 0.5 * (sv_edges[1:] + sv_edges[:-1])
    Dq_left_sum = np.zeros((int(sv_bins), q_grid.size), dtype=float)
    Dq_right_sum = np.zeros((int(sv_bins), q_grid.size), dtype=float)
    bin_count = np.zeros(int(sv_bins), dtype=int)

    for _ in range(num_matrices):
        sub_seed = int(rng.integers(0, 2**31 - 1))
        h = S_star * _belinschi_stable_samples(alpha, N, seed=sub_seed)
        d = phi_prime(torch.tensor(h)).cpu().numpy()
        W = w_scale * _scipy_stable_matrix(alpha, (N, N), seed=sub_seed + 1)
        J = d[:, None] * W
        U, s, Vh = np.linalg.svd(J, full_matrices=False)
        U_abs2 = np.abs(U.T) ** 2     # rows = left SVs
        V_abs2 = np.abs(Vh) ** 2      # rows = right SVs
        for qi, q in enumerate(q_grid):
            if abs(float(q) - 1.0) < 1e-12:
                Dq_left = np.ones(int(N))
                Dq_right = np.ones(int(N))
            else:
                I_q_left = (U_abs2 ** float(q)).sum(axis=-1)
                I_q_right = (V_abs2 ** float(q)).sum(axis=-1)
                I_q_left = np.where(I_q_left > 0, I_q_left, np.nan)
                I_q_right = np.where(I_q_right > 0, I_q_right, np.nan)
                Dq_left = -np.log(I_q_left) / ((q - 1.0) * log_N)
                Dq_right = -np.log(I_q_right) / ((q - 1.0) * log_N)
            s_bin = np.digitize(s, sv_edges) - 1
            for k in range(int(N)):
                sb = s_bin[k]
                if 0 <= sb < int(sv_bins):
                    if np.isfinite(Dq_left[k]):
                        Dq_left_sum[sb, qi] += float(Dq_left[k])
                    if np.isfinite(Dq_right[k]):
                        Dq_right_sum[sb, qi] += float(Dq_right[k])
                    if qi == 0:
                        bin_count[sb] += 1

    safe = bin_count > 0
    Dq_left_mean = np.full_like(Dq_left_sum, np.nan)
    Dq_right_mean = np.full_like(Dq_right_sum, np.nan)
    Dq_left_mean[safe] = Dq_left_sum[safe] / bin_count[safe, None]
    Dq_right_mean[safe] = Dq_right_sum[safe] / bin_count[safe, None]

    return {
        "sv_centres": sv_centres,
        "sv_edges": sv_edges,
        "q_grid": q_grid,
        "Dq_left_mean": Dq_left_mean,
        "Dq_right_mean": Dq_right_mean,
        "bin_count": bin_count,
        "q_star": float(q_star),
        "S_star": float(S_star),
        "N": int(N),
        "num_matrices": int(num_matrices),
    }


# ---------------------------------------------------------------------------
# Density-deviation localisation diagnostic.
#
# Compares the analytical SV density (P1, via osw + Belinschi h_alpha) to the
# population-dynamics estimate (P2, RMT.jac_cavity_svd_log_pdf) at fixed pool
# size, on a fine SV grid.  Systematic deviation -- beyond MC noise --
# signals heavy-tail fluctuations of the local resolvent (the Aizenman-
# Molchanov / BG localisation signature, recast at the spectral-density
# level).  See RMT/localisation.md sec. 9J.
# ---------------------------------------------------------------------------


def density_deviation_diagnostic(
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    s_max: float = 6.0,
    num_points: int = 161,
    num_doublings: int = 8,
    num_chis: int = 4,
    quadrature_order: int = 96,
    profile_order: int = 64,
    imag_eps: float = 1e-3,
    seed: Optional[int] = 0,
) -> dict:
    """Density-deviation diagnostic at a single fixed pool size.

    Returns:
      - sv_grid:        common SV axis.
      - rho_thy:        analytical SV density via osw + Belinschi h_alpha.
      - rho_popdyn:     popdyn estimate (mean over num_chis realisations).
      - rho_popdyn_std: popdyn estimate std (across chi realisations).
      - delta_abs:      |rho_popdyn - rho_thy| (signed deviation).
      - delta_signal:   |rho_popdyn - rho_thy| / max(rho_popdyn_std, eps_floor)
                        -- signal-to-noise of the deviation (S/N > 3 ~ real).
      - q_star, S_star: heavy-tailed length-map fixed point.
    """
    profile = jacobian_profile(alpha, sigma_w, phi=phi, sigma_b=sigma_b,
                               seed=seed)

    # Analytical curve
    curve, _ = theoretical_jacobian_sv_curve(
        alpha, sigma_w, phi=phi, sigma_b=sigma_b, profile=profile,
        s_max=s_max, num_points=num_points,
        quadrature_order=quadrature_order, profile_order=profile_order,
        imag_eps=imag_eps,
    )
    sv_grid = curve.singular_values
    rho_thy = curve.singular_density

    # Population-dynamics density on the same grid (skip s = 0)
    rho_popdyn_pos, rho_popdyn_std_pos = population_dynamics_sv_density(
        sv_grid[1:], alpha, sigma_w, phi=phi, sigma_b=sigma_b,
        q_star=profile.q_star, num_doublings=num_doublings,
        num_chis=num_chis, progress=False, seed=seed,
    )
    rho_popdyn = np.concatenate([[0.0], rho_popdyn_pos])
    rho_popdyn_std = np.concatenate([[0.0], rho_popdyn_std_pos])

    delta_abs = np.abs(rho_popdyn - rho_thy)
    # S/N: relative to the cross-chi MC scatter of popdyn.
    noise_floor = 1e-12
    delta_signal = delta_abs / np.maximum(rho_popdyn_std, noise_floor)

    return {
        "sv_grid": sv_grid,
        "rho_thy": rho_thy,
        "rho_popdyn": rho_popdyn,
        "rho_popdyn_std": rho_popdyn_std,
        "delta_abs": delta_abs,
        "delta_signal": delta_signal,
        "q_star": profile.q_star,
        "S_star": profile.S_star,
        "alpha": float(alpha),
        "sigma_w": float(sigma_w),
        "num_doublings": int(num_doublings),
        "num_chis": int(num_chis),
        "pool_size": int(2 ** num_doublings),
    }


# ---------------------------------------------------------------------------
# Pool-size scaling sweep -- heavy-tail exponent nu(s) of the local
# resolvent distribution.
#
# Idea: at each SV s, run popdyn at multiple pool sizes P = 2^{n_d} and
# track |Delta_rho(s; P)| = |rho_popdyn(s; P) - rho_thy(s)|.  Fit
# log|Delta| vs log P; slope m maps to the heavy-tail index nu of the
# local resolvent via the generalised CLT:
#
#   nu > 2:  CLT applies, |Delta| ~ P^{-1/2}.  Slope m = -1/2 -> nu = 2.
#   1 < nu < 2:  heavy-tail CLT, |Delta| ~ P^{(1/nu) - 1}.
#                Slope m in (-1/2, 0) -> nu = 1/(m + 1) in (1, 2).
#   nu < 1:  pool-mean diverges, |Delta| ~ P^{(1/nu) - 1}, slope m > 0.
#                nu = 1/(m + 1).
#
# nu < 1 = strongly localised (pool-mean fails to converge).
# 1 < nu < 2 = heavy-tail / partial localisation.
# nu >= 2 = delocalised (no signature in this diagnostic).
# ---------------------------------------------------------------------------


def density_deviation_pool_sweep(
    alpha: float,
    sigma_w: float,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    s_max: float = 5.0,
    num_points: int = 81,
    pool_doublings_list: tuple = (4, 5, 6, 7, 8),
    num_chis: int = 4,
    quadrature_order: int = 96,
    profile_order: int = 64,
    imag_eps: float = 1e-3,
    seed: Optional[int] = 0,
    num_steps_per_element: Optional[int] = 100,
) -> dict:
    """Run popdyn at increasing pool sizes; fit log|Delta_rho| vs log P
    at each SV to extract the heavy-tail exponent nu(s) of the local
    resolvent distribution.

    ``num_steps_per_element``: passes
    ``num_steps_fn = lambda P: int(num_steps_per_element * P)`` to popdyn,
    so each pool element is updated ~num_steps_per_element times regardless
    of pool size.  This DECOUPLES mixing (iterations-per-element) from
    pool-mean MC noise.  Default 100 is large enough that the pool is well-
    mixed at all sizes in [16, 256].  Set to None to use RMT.py's default
    P**2 scaling (which couples mixing to pool size; bulk slopes then
    reflect iteration-rate not pool-mean CLT).

    Returns:
      - sv_grid, rho_thy, rho_popdyn_by_P, delta_by_P, slopes, nu, r_squared, Ps.
    """
    profile = jacobian_profile(alpha, sigma_w, phi=phi, sigma_b=sigma_b,
                               seed=seed)

    curve, _ = theoretical_jacobian_sv_curve(
        alpha, sigma_w, phi=phi, sigma_b=sigma_b, profile=profile,
        s_max=s_max, num_points=num_points,
        quadrature_order=quadrature_order, profile_order=profile_order,
        imag_eps=imag_eps,
    )
    sv_grid = curve.singular_values
    rho_thy = curve.singular_density

    if num_steps_per_element is not None:
        c = int(num_steps_per_element)
        num_steps_fn = lambda P: int(c * P)  # noqa: E731
    else:
        num_steps_fn = None

    rho_popdyn_by_P = {}
    for n_d in pool_doublings_list:
        P = int(2 ** int(n_d))
        rho_pos, _ = population_dynamics_sv_density(
            sv_grid[1:], alpha, sigma_w, phi=phi, sigma_b=sigma_b,
            q_star=profile.q_star, num_doublings=int(n_d),
            num_chis=num_chis, progress=False, seed=seed,
            num_steps_fn=num_steps_fn,
        )
        rho_popdyn_by_P[P] = np.concatenate([[0.0], rho_pos])

    delta_by_P = {P: np.abs(r - rho_thy) for P, r in rho_popdyn_by_P.items()}

    Ps = sorted(rho_popdyn_by_P.keys())
    log_Ps = np.log(np.asarray(Ps, dtype=float))
    slopes = np.full(int(num_points), np.nan, dtype=float)
    intercepts = np.full(int(num_points), np.nan, dtype=float)
    r_squared = np.full(int(num_points), np.nan, dtype=float)
    for i in range(1, int(num_points)):
        log_deltas = np.array([
            np.log(max(delta_by_P[P][i], 1e-15)) for P in Ps
        ])
        if np.all(np.isfinite(log_deltas)):
            # weighted least-squares slope
            slope, intercept = np.polyfit(log_Ps, log_deltas, 1)
            slopes[i] = slope
            intercepts[i] = intercept
            # R^2
            pred = slope * log_Ps + intercept
            ss_res = np.sum((log_deltas - pred) ** 2)
            ss_tot = np.sum((log_deltas - log_deltas.mean()) ** 2)
            r_squared[i] = 1.0 - ss_res / max(ss_tot, 1e-15)

    nu = 1.0 / (slopes + 1.0)

    return {
        "sv_grid": sv_grid,
        "rho_thy": rho_thy,
        "rho_popdyn_by_P": rho_popdyn_by_P,
        "delta_by_P": delta_by_P,
        "slopes": slopes,
        "nu": nu,
        "r_squared": r_squared,
        "Ps": np.asarray(Ps),
        "alpha": float(alpha),
        "sigma_w": float(sigma_w),
        "num_chis": int(num_chis),
    }


# ---------------------------------------------------------------------------
# Verify the analytical D_q formula (eq. 14 of RMT/localisation.md) against
# direct SVD-IPR on the same matrix realisations.
#
# Formula:   D_q = 1 - log[M_q / (pi rho)^q] / ((q-1) log N)
#            M_q = (1/N) sum_i (-Im G_ii(lambda + i eta))^q
# computed from the bipartite resolvent spectral decomposition:
#   -Im G^(1)_ii(lambda + i eta) = (1/2) sum_k |u_k(i)|^2 eta / ((lambda - s_k)^2 + eta^2)
# (using only positive-s_k contributions for lambda > 0; the negative-s
# half-spectrum contribution is negligible there).  Same u_k, s_k as in
# the direct IPR computation -- the two estimators come from the same
# matrices.  Agreement verifies the BG self-averaging identity (eq. 12)
# and hence the formula.
# ---------------------------------------------------------------------------


def verify_Dq_formula(
    alpha: float,
    sigma_w: float,
    q_grid: np.ndarray,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    N: int = 256,
    depth: int = 50,
    num_matrices: int = 10,
    burn_in: int = 25,
    eta: Optional[float] = None,
    seed: Optional[int] = None,
    sv_bins: int = 21,
    sv_range: Optional[tuple[float, float]] = None,
) -> dict:
    """Verify eq. (14) of localisation.md against direct SVD IPR.

    For each MLP realisation:
    - Run forward with compute_uv=True to get s, u, v at post-burn-in layers.
    - For each layer, take lambda = s_k for each k.  Compute:
       (a) IPR direct:    I_q^(k) = sum_i |u_k(i)|^{2q} (and same for v).
       (b) Im G^(1)_ii:   from the spectral decomposition (formula above)
                          at lambda = s_k, smoothing parameter eta.
                          M_q = (1/N) sum_i (-Im G_ii)^q.
                          rho = (1/N pi) sum_i (-Im G_ii).
                          D_q^formula = 1 - log[M_q/(pi rho)^q]/((q-1) log N).
    - Bin (k -> SV bin), average.  Return both estimators.

    eta defaults to 1/N (level-spacing scale).
    """
    q_grid = np.asarray(q_grid, dtype=float)
    rng = np.random.default_rng(seed)
    if eta is None:
        eta = 1.0 / float(N)
    log_N = float(np.log(max(int(N), 2)))

    # Pre-scan if needed
    if sv_range is None:
        x0 = rng.standard_normal(N).astype(np.float32)
        sub = int(rng.integers(0, 2**31 - 1))
        st = RMT.MLP(x0, depth=depth, alpha=alpha, sigma_W=sigma_w,
                     sigma_b=sigma_b, phi=phi, seed=sub,
                     fast=False, compute_uv=False, usetqdm=False)
        log_sv = st["postjac_log_svdvals"][burn_in:]
        sv_range = (0.0, float(np.percentile(np.exp(log_sv).reshape(-1), 99.0)))

    sv_edges = np.linspace(sv_range[0], sv_range[1], int(sv_bins) + 1)
    sv_centres = 0.5 * (sv_edges[1:] + sv_edges[:-1])

    # Per-bin accumulators
    Dq_ipr_left_sum = np.zeros((int(sv_bins), q_grid.size))
    Dq_ipr_right_sum = np.zeros((int(sv_bins), q_grid.size))
    Dq_form_left_sum = np.zeros((int(sv_bins), q_grid.size))
    Dq_form_right_sum = np.zeros((int(sv_bins), q_grid.size))
    bin_count = np.zeros(int(sv_bins), dtype=int)

    for _ in range(num_matrices):
        x0 = rng.standard_normal(N).astype(np.float32)
        sub = int(rng.integers(0, 2**31 - 1))
        st = RMT.MLP(x0, depth=depth, alpha=alpha, sigma_W=sigma_w,
                     sigma_b=sigma_b, phi=phi, seed=sub,
                     fast=False, compute_uv=True, usetqdm=False)
        log_sv = st["postjac_log_svdvals"][burn_in:]      # (L', N)
        u_layers = st["postjac_svd_left"][burn_in:]        # (L', N, N) rows=u_k
        v_layers = st["postjac_svd_right"][burn_in:]       # (L', N, N) rows=v_k

        for ell in range(log_sv.shape[0]):
            sv_vals = np.exp(log_sv[ell])                  # (N,)
            U_abs2 = np.abs(u_layers[ell]) ** 2            # (N, N)  rows=u_k
            V_abs2 = np.abs(v_layers[ell]) ** 2            # (N, N)  rows=v_k

            # For each k, lambda = s_k.  Compute Im G^(1)_ii(s_k + i eta) for each i.
            # Im G^(1)_ii = (1/2) sum_k' |u_{k'}(i)|^2 eta / ((s_k - s_{k'})^2 + eta^2).
            # Vectorise over (k, k').  Cost O(N^2 N) = O(N^3) per layer; OK at N=256.
            ds = sv_vals[:, None] - sv_vals[None, :]       # (N, N) (k, k')
            lorentz = eta / (ds ** 2 + eta ** 2)            # (N, N)
            # Im G_ii for type-1 (row, left SVs):
            # ImG^(1)_{ii}[k, i] = (1/2) sum_{k'} U_abs2[k', i] * lorentz[k, k']
            ImG_left = 0.5 * lorentz @ U_abs2               # (N, N): rows index k (lambda), cols index i
            ImG_right = 0.5 * lorentz @ V_abs2

            rho_left = ImG_left.mean(axis=-1) / np.pi       # (N,), rho^(1)(s_k)
            rho_right = ImG_right.mean(axis=-1) / np.pi

            # M_q for each k:
            # IPR direct:
            for qi, q in enumerate(q_grid):
                if abs(float(q) - 1.0) < 1e-12:
                    Dq_ipr_left = np.ones(int(N))
                    Dq_ipr_right = np.ones(int(N))
                    Dq_form_left = np.ones(int(N))
                    Dq_form_right = np.ones(int(N))
                else:
                    I_q_left = (U_abs2 ** float(q)).sum(axis=-1)
                    I_q_right = (V_abs2 ** float(q)).sum(axis=-1)
                    I_q_left = np.where(I_q_left > 0, I_q_left, np.nan)
                    I_q_right = np.where(I_q_right > 0, I_q_right, np.nan)
                    Dq_ipr_left = -np.log(I_q_left) / ((q - 1.0) * log_N)
                    Dq_ipr_right = -np.log(I_q_right) / ((q - 1.0) * log_N)

                    M_q_left = (ImG_left ** float(q)).mean(axis=-1)
                    M_q_right = (ImG_right ** float(q)).mean(axis=-1)
                    ratio_l = M_q_left / np.maximum((np.pi * rho_left) ** float(q), 1e-300)
                    ratio_r = M_q_right / np.maximum((np.pi * rho_right) ** float(q), 1e-300)
                    ok_l = (ratio_l > 0) & np.isfinite(ratio_l) & (rho_left > 0)
                    ok_r = (ratio_r > 0) & np.isfinite(ratio_r) & (rho_right > 0)
                    Dq_form_left = np.full(int(N), np.nan)
                    Dq_form_right = np.full(int(N), np.nan)
                    Dq_form_left[ok_l] = 1.0 - np.log(ratio_l[ok_l]) / ((q - 1.0) * log_N)
                    Dq_form_right[ok_r] = 1.0 - np.log(ratio_r[ok_r]) / ((q - 1.0) * log_N)
                s_bin = np.digitize(sv_vals, sv_edges) - 1
                for k in range(int(N)):
                    sb = s_bin[k]
                    if 0 <= sb < int(sv_bins):
                        if np.isfinite(Dq_ipr_left[k]):
                            Dq_ipr_left_sum[sb, qi] += float(Dq_ipr_left[k])
                        if np.isfinite(Dq_ipr_right[k]):
                            Dq_ipr_right_sum[sb, qi] += float(Dq_ipr_right[k])
                        if np.isfinite(Dq_form_left[k]):
                            Dq_form_left_sum[sb, qi] += float(Dq_form_left[k])
                        if np.isfinite(Dq_form_right[k]):
                            Dq_form_right_sum[sb, qi] += float(Dq_form_right[k])
                        if qi == 0:
                            bin_count[sb] += 1

    safe = bin_count > 0
    def _avg(arr):
        out = np.full_like(arr, np.nan)
        out[safe] = arr[safe] / bin_count[safe, None]
        return out

    return {
        "sv_centres": sv_centres,
        "sv_edges": sv_edges,
        "q_grid": q_grid,
        "eta": float(eta),
        "Dq_ipr_left": _avg(Dq_ipr_left_sum),
        "Dq_ipr_right": _avg(Dq_ipr_right_sum),
        "Dq_form_left": _avg(Dq_form_left_sum),
        "Dq_form_right": _avg(Dq_form_right_sum),
        "bin_count": bin_count,
        "N": int(N),
        "num_matrices": int(num_matrices),
    }


# ---------------------------------------------------------------------------
# Convention check (advisor flag): structured-curve sampling vs MLP-style
# Belinschi sampling on a constant profile (no phi' -- isolates the conv).
# ---------------------------------------------------------------------------


def convention_check(
    alpha: float = 1.5,
    sigma_w: float = 1.0,
    *,
    N: int = 256,
    num_matrices: int = 80,
    bins: int = 121,
    seed: Optional[int] = 0,
) -> dict:
    """Sanity gate: empirical SV histogram from `sample_structured_levy_matrix`
    (with constant tau=1, entry_scale=sigma_w) should match the histogram of
    `sigma_w * (2N)^{-1/alpha} * Belinschi-stable` matrices -- the
    convention used by `RMT.py:MLP` to draw weights.

    Returns max abs density diff on the common support.  Pass if < ~5e-3 at
    N=256, num_matrices=80; the threshold is set by Monte-Carlo bin noise,
    not the convention.
    """
    rng = np.random.default_rng(seed)

    # Route A: structured-curve native sampler (no nonlinearity, tau=1).
    sv_a = []
    for _ in range(num_matrices):
        M, *_ = swl.sample_structured_levy_matrix(
            n_rows=N, alpha=alpha, gamma=1.0, tau=1.0,
            entry_scale=sigma_w, normalization="stable",
            random_state=rng,
        )
        sv_a.append(np.linalg.svd(M, compute_uv=False))
    sv_a = np.concatenate(sv_a)

    # Route B: MLP-style W sampling (sigma_w * (2N)^{-1/alpha} * Belinschi).
    rng_b = np.random.default_rng(seed)
    sv_b = []
    w_scale = sigma_w * (2.0 * N) ** (-1.0 / alpha)
    for _ in range(num_matrices):
        sub_seed = int(rng_b.integers(0, 2**31 - 1))
        W = w_scale * _scipy_stable_matrix(alpha, (N, N), seed=sub_seed)
        sv_b.append(np.linalg.svd(W, compute_uv=False))
    sv_b = np.concatenate(sv_b)

    s_max = float(np.percentile(np.concatenate([sv_a, sv_b]), 99.0))
    dens_a, edges = np.histogram(sv_a, bins=bins, range=(0.0, s_max), density=True)
    dens_b, _ = np.histogram(sv_b, bins=bins, range=(0.0, s_max), density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    return {
        "centres": centres,
        "density_structured_native": dens_a,
        "density_mlp_style": dens_b,
        "max_abs_diff": float(np.max(np.abs(dens_a - dens_b))),
        "rel_L1_diff": float(
            np.sum(np.abs(dens_a - dens_b)) * (edges[1] - edges[0])
            / max(np.sum(dens_a) * (edges[1] - edges[0]), 1e-12)
        ),
        "s_max": s_max,
    }


# ---------------------------------------------------------------------------
# Full three-way validation.
# ---------------------------------------------------------------------------


def run_validation(
    alpha: float = 1.5,
    sigma_W: float = 1.0,
    *,
    phi: Callable = torch.tanh,
    sigma_b: float = 0.0,
    N: int = 256,
    num_matrices: int = 60,
    depth: int = 60,
    burn_in: int = 25,
    num_doublings: int = 8,
    num_chis: int = 1,
    n_profile_samples: int = 50_000,
    s_max: float = 8.0,
    num_points: int = 161,
    bins: int = 121,
    seed: Optional[int] = 0,
    save_path: Optional[str] = None,
    do_mlp: bool = True,
) -> dict:
    """End-to-end three-way (four-way) validation at a single (alpha, sigma_w).

    Returns a dict of arrays suitable for plotting; saves to .npz if
    save_path is given.  Times every stage.
    """
    timings: list = []

    with Timer("convention_check", timings):
        conv = convention_check(alpha=alpha, sigma_w=sigma_W, N=N,
                                num_matrices=min(num_matrices, 40),
                                bins=bins, seed=seed)

    with Timer("jacobian_profile", timings):
        profile = jacobian_profile(alpha, sigma_W, phi=phi, sigma_b=sigma_b,
                                   n_samples=n_profile_samples, seed=seed)

    with Timer("(P1) theoretical curve", timings):
        curve, _ = theoretical_jacobian_sv_curve(
            alpha, sigma_W, phi=phi, sigma_b=sigma_b,
            s_max=s_max, num_points=num_points,
            profile=profile,
        )
    B = theoretical_tail_constant(profile)

    with Timer("(P2) population dynamics", timings):
        # The cavity engine wants the SV grid; reuse the structured grid (P1).
        # Skip s=0 (cavity has a log-density there).
        sv_grid_p2 = curve.singular_values[1:]
        pop_density, pop_density_std = population_dynamics_sv_density(
            sv_grid_p2, alpha, sigma_W, phi=phi, sigma_b=sigma_b,
            q_star=profile.q_star, num_doublings=num_doublings,
            num_chis=num_chis, progress=False, seed=seed,
        )

    with Timer("(P3a) synthetic empirical", timings):
        emp_syn = synthetic_jacobian_sv_spectrum(
            alpha, sigma_W, phi=phi, sigma_b=sigma_b, q_star=profile.q_star,
            N=N, num_matrices=num_matrices, seed=seed, bins=bins,
            sv_range=(0.0, s_max),
        )

    emp_mlp = None
    if do_mlp:
        with Timer("(P3b) MLP-derived empirical", timings):
            emp_mlp = mlp_jacobian_sv_spectrum(
                alpha, sigma_W, phi=phi, sigma_b=sigma_b, q_star=profile.q_star,
                N=N, depth=depth, num_matrices=max(1, num_matrices // 4),
                burn_in=burn_in, seed=seed, bins=bins, sv_range=(0.0, s_max),
            )

    # ---- agreement metrics on a common SV grid (P3a histogram centres) ----
    # The structured curve hard-sets singular_density[0] = 0 even at gamma=1
    # where the true density is nonzero (Wigner/plain-square corner of
    # structured_wishart_levy.md "Specializations" remark).  Skip the s=0
    # boundary point in the interp so the first bin isn't biased by that 0.
    sv_theory_pos = curve.singular_values[1:]
    den_theory_pos = curve.singular_density[1:]
    centres = emp_syn["centres"]
    theory_on_centres = np.interp(centres, sv_theory_pos, den_theory_pos)
    p2_on_centres = np.interp(centres, sv_grid_p2, pop_density)
    # Also restrict comparison to s above one bin (avoids the smallest-s
    # boundary cusp where finite-N effects in (P3a) and the imag_eps
    # smoothing in (P1) compete).
    bin_step = float(centres[1] - centres[0]) if len(centres) > 1 else 0.0
    cmp_mask = centres > 2 * bin_step
    diff_theory_syn = float(np.max(np.abs(
        theory_on_centres[cmp_mask] - emp_syn["density"][cmp_mask])))
    diff_p2_syn = float(np.max(np.abs(
        p2_on_centres[cmp_mask] - emp_syn["density"][cmp_mask])))
    diff_theory_p2 = float(np.max(np.abs(
        theory_on_centres[cmp_mask] - p2_on_centres[cmp_mask])))

    summary = {
        "alpha": float(alpha),
        "sigma_W": float(sigma_W),
        "q_star": profile.q_star,
        "S_star": profile.S_star,
        "profile_alpha_moment": profile.profile_alpha_moment,
        "tail_constant_B": float(B),
        "convention_max_abs_diff": float(conv["max_abs_diff"]),
        "convention_rel_L1": float(conv["rel_L1_diff"]),
        "diff_theory_synthetic": diff_theory_syn,
        "diff_pop_dyn_synthetic": diff_p2_syn,
        "diff_theory_pop_dyn": diff_theory_p2,
        "q_rel_err_mlp": float(emp_mlp["q_rel_err"]) if emp_mlp else float("nan"),
        "timings": dict(timings),
    }
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = {
        "summary": summary,
        "sv_theory": curve.singular_values,
        "density_theory": curve.singular_density,
        "sv_grid_pop_dyn": sv_grid_p2,
        "density_pop_dyn": pop_density,
        "density_pop_dyn_std": pop_density_std,
        "sv_synthetic_centres": emp_syn["centres"],
        "density_synthetic": emp_syn["density"],
        "sv_mlp_centres": (emp_mlp["centres"] if emp_mlp else np.array([])),
        "density_mlp": (emp_mlp["density"] if emp_mlp else np.array([])),
        "convention_centres": conv["centres"],
        "convention_density_structured": conv["density_structured_native"],
        "convention_density_mlp_style": conv["density_mlp_style"],
        "profile_samples": profile.profile_samples,
    }
    if save_path is not None:
        np.savez(save_path, **{k: v for k, v in out.items() if k != "summary"})
        print(f"\nsaved -> {save_path}")
    return out


# ---------------------------------------------------------------------------
# CLI (RMT.py-style: keyword args with explicit type conversion).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    tic = time()
    func_name = sys.argv[1]
    func = {
        "run_validation": run_validation,
        "convention_check": convention_check,
        "theoretical_jacobian_sv_curve": theoretical_jacobian_sv_curve,
        "synthetic_jacobian_sv_spectrum": synthetic_jacobian_sv_spectrum,
        "mlp_jacobian_sv_spectrum": mlp_jacobian_sv_spectrum,
        "jacobian_profile": jacobian_profile,
    }[func_name]
    args = [sys.argv[i : i + 3] for i in range(2, len(sys.argv), 3)]
    arg_dict = {arg[0]: eval(arg[2])(arg[1]) for arg in args}
    result = func(**arg_dict)
    if result is not None and func_name != "run_validation":
        print(result)
    print(f"\n[total] {time() - tic:.2f}s")
