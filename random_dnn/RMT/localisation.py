"""Localisation observables for structured heavy-tailed random matrices.

Companion to ``RMT/localisation.md``.  Currently implements:

**Profile-aligned localisation index ell_q** (sec. 2-8 of the md, all
derived).  Jensen-gap of the deterministic per-position mean LDoS,
detects profile-axis asymmetry of singular vectors via the
deterministic field Y_r(x), Y_c(y) (one-sided: Y_r constant in x, all
action on column side via slaved Y_c(y)).

**Not implemented: D_q (multifractal exponent).** Sec. 9 of the md
sketches two formulations (via E|G|^q and E(-Im G)^q).  A 2-D-pushforward
prototype was written here but is fundamentally limited -- Belinschi's
1-scalar CF parameterisation does not determine the full 2-D distribution
of Sigma needed to evaluate the pushforward.  See the "Part 2" placeholder
in this module and sec. 9 of ``RMT/localisation.md`` for the status and
list of paths forward.

DAG: ``localisation -> {one_sided_wishart_levy, structured_wishart_levy,
wishart_levy}``.  Acyclic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import special

import wishart_levy as wl
import structured_wishart_levy as swl
import one_sided_wishart_levy as osw

# Re-export the profile spec for callers
OneSidedProfileSpec = osw.OneSidedProfileSpec


# ===========================================================================
# Part 1: profile-aligned localisation index ell_q
# (Moved from one_sided_wishart_levy.py)
# ===========================================================================


@dataclass
class OneSidedLocalisationCurve:
    alpha: float
    gamma: float
    q: float
    profile_name: str
    singular_values: np.ndarray            # shape (num_points,)
    ell_q_col: np.ndarray                  # shape (num_points,), in [0, 1]
    per_column_ldos: np.ndarray            # shape (num_points, profile_order)
    column_nodes: np.ndarray               # Gauss-Legendre nodes on [0, 1]
    column_weights: np.ndarray             # corresponding weights
    column_profile_values: np.ndarray      # c(y_nodes)


def _per_column_ldos_from_curve(
    curve: osw.OneSidedTheoryCurve,
    *,
    c: OneSidedProfileSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the per-column LDoS rho^{(2)}_y(s) on the curve's SV grid.

    Uses the slaved column-field formula (eq. 8 of ``RMT/localisation.md``)
    Y_c(y, z) = C_alpha c(y)^alpha g_alpha(Y_r(z)) / ((1 + gamma) z^alpha),
    then independent-rule h_alpha for the readout (matches the SV-density
    convention in one_sided_wishart_levy).

    Returns (rho_per_col, y_nodes, y_weights, c_vals) where rho_per_col has
    shape (num_points, profile_order).
    """
    alpha = curve.alpha
    gamma = curve.gamma
    quadrature_order = curve.quadrature_order
    profile_order = curve.profile_order
    output_scale = wl._output_scale(alpha, curve.entry_scale, curve.normalization)
    C_a = wl.belinschi_constant(alpha)

    cb, _ = osw._profile_callable(c)
    y_nodes, y_w = swl._legendre01(int(profile_order))
    c_vals = np.asarray(cb(y_nodes), dtype=float)
    c_alpha_arr = np.abs(c_vals) ** alpha

    sv = curve.singular_values
    rho_per_col = np.zeros((sv.size, c_vals.size), dtype=float)
    for idx in range(1, sv.size):
        y_r = curve.y_row[idx]
        if not np.isfinite(y_r):
            rho_per_col[idx, :] = np.nan
            continue
        s_out = sv[idx]
        s_base = s_out / output_scale
        z = complex(s_base, curve.imag_eps)
        z_alpha = z ** alpha
        g_r = complex(swl._g_alpha_vec(np.array([y_r]), alpha, quadrature_order)[0])
        Y_c_vec = (C_a * c_alpha_arr / ((1.0 + gamma) * z_alpha)) * g_r
        rho_y = np.empty(c_vals.size, dtype=float)
        for j, Y_c_j in enumerate(Y_c_vec):
            h_val = osw._h_alpha_indep(complex(Y_c_j), alpha, quadrature_order)
            rho_y[j] = max(0.0, -h_val.imag / (np.pi * s_base)) / output_scale
        rho_per_col[idx, :] = rho_y
    return rho_per_col, y_nodes, y_w, c_vals


def _localisation_index_from_ldos(
    rho_per_col: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> np.ndarray:
    """ell_q(s) = 1 - M_q(s) / bar_rho(s)^q via Gauss-Legendre integration.

    rho_per_col: shape (num_points, n_y).  weights: shape (n_y,), sums to 1.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must satisfy 0 < q < 1 (the Jensen-concave range "
                         "for the localisation index; see localisation.md sec. 2).")
    bar_rho = rho_per_col @ weights
    M_q = (rho_per_col ** q) @ weights
    safe = bar_rho > 0.0
    ell = np.zeros_like(bar_rho)
    ell[safe] = 1.0 - M_q[safe] / (bar_rho[safe] ** q)
    ell[~safe] = np.nan
    return np.clip(ell, 0.0, 1.0)


def localisation_index_curve(
    alpha: float,
    gamma: float,
    c: OneSidedProfileSpec,
    *,
    q: Optional[float] = None,
    curve: Optional[osw.OneSidedTheoryCurve] = None,
    s_max: float = 8.0,
    num_points: int = 161,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    profile_order: int = 64,
    tol: float = 1e-12,
) -> OneSidedLocalisationCurve:
    """Column-side profile-aligned localisation index ell_q^{(2)}(s).

    Specialisation of ``RMT/localisation.md`` to the one-sided
    ``|tau(x, y)| = c(y)`` case (sec. 8): the row index ell_q^{(1)} is
    identically zero, all profile-aligned localisation lives on the
    column side via the slaved Y_c(y, z).

    If ``curve`` is given the field is reused (no extra Theorem 2 solve);
    otherwise it is computed.  Default ``q = alpha / 2`` parallels
    Bordenave-Guionnet; restrict to ``0 < q < 1`` so that Jensen's
    concavity applies (see localisation.md sec. 2).
    """
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    if q is None:
        q = alpha / 2.0
    if curve is None:
        curve = osw.theoretical_one_sided_singular_value_curve(
            alpha, gamma, c, s_max=s_max, num_points=num_points,
            entry_scale=entry_scale, normalization=normalization,
            imag_eps=imag_eps, quadrature_order=quadrature_order,
            profile_order=profile_order, tol=tol,
        )
    rho_per_col, y_nodes, y_w, c_vals = _per_column_ldos_from_curve(curve, c=c)
    ell = _localisation_index_from_ldos(rho_per_col, y_w, float(q))
    return OneSidedLocalisationCurve(
        alpha=float(alpha), gamma=float(gamma), q=float(q),
        profile_name=curve.profile_name,
        singular_values=curve.singular_values,
        ell_q_col=ell,
        per_column_ldos=rho_per_col,
        column_nodes=y_nodes,
        column_weights=y_w,
        column_profile_values=c_vals,
    )


def empirical_localisation_index_from_svd(
    alpha: float,
    gamma: float,
    c: OneSidedProfileSpec,
    *,
    q: Optional[float] = None,
    n_rows: int = 400,
    num_matrices: int = 60,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    sv_bins: int = 21,
    sv_range: Optional[tuple[float, float]] = None,
    seed: Optional[int] = None,
    n_col_bins: Optional[int] = None,
) -> dict:
    """Empirical ell_q^{(2)}(s) from SVD right singular vectors.

    For each sampled matrix, computes |v_k(j)|^2 (squared right-SV component
    at column j) and accumulates into a 2-D histogram binned by (SV, column
    position).  The Jensen gap (eq. 3 of localisation.md, sec. 2) is then
    evaluated bin-by-bin on the smoothed per-column LDoS.

    ``n_col_bins`` defaults to ceil(sqrt(n_cols)) -- coarser than per-column
    so the per-bin mass is large enough to estimate the q-moment reliably.
    """
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    if q is None:
        q = alpha / 2.0
    if not (0.0 < q < 1.0):
        raise ValueError("q must satisfy 0 < q < 1.")
    rng = np.random.default_rng(seed)
    n_cols = max(1, int(round(gamma * n_rows)))
    if n_col_bins is None:
        n_col_bins = max(8, int(np.ceil(np.sqrt(n_cols))))

    if sv_range is None:
        M0, _y, _c = osw.sample_one_sided_levy_matrix(
            n_rows, alpha, gamma, c, entry_scale=entry_scale,
            normalization=normalization, random_state=rng,
        )
        s0 = np.linalg.svd(M0, compute_uv=False)
        sv_range = (0.0, float(np.percentile(s0, 99.0)))

    sv_edges = np.linspace(sv_range[0], sv_range[1], int(sv_bins) + 1)
    sv_centres = 0.5 * (sv_edges[1:] + sv_edges[:-1])
    col_edges = np.linspace(0.0, 1.0, int(n_col_bins) + 1)
    col_centres = 0.5 * (col_edges[1:] + col_edges[:-1])

    hist = np.zeros((sv_bins, n_col_bins), dtype=float)
    sv_count = np.zeros(sv_bins, dtype=float)
    j_grid = (np.arange(n_cols) + 0.5) / n_cols
    j_bin = np.clip(np.digitize(j_grid, col_edges) - 1, 0, n_col_bins - 1)

    for _ in range(num_matrices):
        M, _, _ = osw.sample_one_sided_levy_matrix(
            n_rows, alpha, gamma, c, entry_scale=entry_scale,
            normalization=normalization, random_state=rng,
        )
        _, s, vh = np.linalg.svd(M, full_matrices=False)
        amp_sq = np.abs(vh) ** 2
        s_bin = np.digitize(s, sv_edges) - 1
        valid = (s_bin >= 0) & (s_bin < sv_bins)
        for k in np.where(valid)[0]:
            sb = s_bin[k]
            np.add.at(hist[sb], j_bin, amp_sq[k])
            sv_count[sb] += 1.0

    safe = sv_count > 0
    ldos = np.zeros_like(hist)
    ldos[safe] = hist[safe] / sv_count[safe, None] / (1.0 / n_col_bins)

    weights = np.full(n_col_bins, 1.0 / n_col_bins)
    bar_rho = ldos @ weights
    M_q = (ldos ** q) @ weights
    ell = np.zeros(sv_bins, dtype=float)
    pos = bar_rho > 0.0
    ell[pos] = 1.0 - M_q[pos] / (bar_rho[pos] ** q)
    ell = np.clip(ell, 0.0, 1.0)
    ell[~pos] = np.nan
    return {
        "sv_centres": sv_centres,
        "sv_edges": sv_edges,
        "column_bin_centres": col_centres,
        "per_column_ldos": ldos,
        "ell_q_col": ell,
        "sv_counts": sv_count,
        "q": float(q),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_col_bins": int(n_col_bins),
        "num_matrices": int(num_matrices),
    }


# ===========================================================================
# Part 2: D_q (multifractal exponent) -- NOT IMPLEMENTED
# ===========================================================================
#
# The 2-D-pushforward scheme sketched in sec. 9F of RMT/localisation.md
# was prototyped here but is fundamentally limited: Belinschi's `cocott`
# formula parameterises P^{mu^z_r} through a single complex scalar X_r,
# which determines the LT/CF only along a one-parameter slice of the 2-D
# Fourier domain, not the full 2-D distribution.  Recovering the density
# of Sigma on C^- via 2-D inverse FFT therefore requires the full 2-D
# CF, which BAG/Belinschi do not establish from Y_r alone (BAG line 405:
# "we cannot prove uniqueness of the solution to this equation" for the
# full mu^z_r).  Numerically the symptom is that the inverse FFT of the
# 1-slice CF puts non-trivial mass on Im Sigma > 0, contradicting the
# C^- support property.
#
# Paths forward, deferred until a path is selected:
#   (1) Closed-form analytic moments only: E[G^q] = g_{alpha, 2q}(Y_r);
#       gives complex moments at any q, NOT the IPR moments.
#   (2) Rotationally-symmetric ansatz on mu^z_r (Cizeau-Bouchaud style)
#       and proceed -- non-rigorous.
#   (3) Population dynamics on the cavity RDE.  Valid for convergent
#       (q < q_c) moments; pool extremes don't extrapolate to physical
#       N above q_c (the user's stated objection).
#   (4) Additional self-consistency equations beyond X_r: e.g., for
#       int |x|^{alpha/2} dmu^z, int x dmu^z, etc.  New derivation
#       work; not in the literature I have notes for.
#   (5) Empirical-only D_q from MLP-Jacobian SVDs via RMT.MLP_agg.
#
# See RMT/localisation.md sec. 9 for the full status discussion.


# ===========================================================================
# D_q via the cavity-RDE population dynamics (sec. 9H of localisation.md)
# ===========================================================================
#
# Formula (eq. 14 of localisation.md):
#     D_q(lambda) = 1 - log[M_q / (pi rho)^q] / ((q - 1) log N)
#     M_q = E_{mu^z}[(-Im G)^q]
# with mu^z the BAG/Belinschi cavity-resolvent distribution.  mu^z is
# realised numerically via the cavity RDE (RMT.cavity_svd_resolvent),
# which maintains pool samples representing mu^z_r (row side, left SVs)
# and mu^z_c (column side, right SVs).


def theoretical_Dq_curve_popdyn(
    alpha: float,
    sigma_w: float,
    q_grid: np.ndarray,
    sv_grid: np.ndarray,
    *,
    phi: object = None,
    sigma_b: float = 0.0,
    q_star: Optional[float] = None,
    N_reference: int = 256,
    num_doublings: int = 7,
    num_chis: int = 1,
    seed: Optional[int] = None,
) -> dict:
    """Theoretical D_q on the row (left) and column (right) sides, with
    the cavity-ensemble moment M_q obtained from population dynamics on
    the BAG cavity RDE.

    Sec. 9H of RMT/localisation.md.  Implements eq. (14):
        D_q = 1 - log[M_q / (pi rho)^q] / ((q - 1) log N_reference)
    with M_q = E_{mu^z}[(-Im G)^q] read off from the converged cavity
    pool produced by RMT.cavity_svd_resolvent.

    Parameters
    ----------
    sigma_w, alpha, phi, sigma_b
        Heavy-tailed MLP parameters; phi defaults to torch.tanh.  Used to
        build the chi-profile chi_j = sigma_w |phi'((q*)^{1/a} z_j)|,
        z_j ~ p_alpha, matching the structured-Wishart-Levy column
        profile c(v) = F^{-1}_{|phi'(S* Z)|}(v) of ht_mlp_jacobian.md.
    q_grid : array of q values for D_q.
    sv_grid : array of singular values lambda for D_q(lambda).
    N_reference : the physical N for the finite-N D_q formula.
    num_doublings, num_chis : passed to RMT.cavity_svd_resolvent.
    """
    # late imports to keep DAG light
    import sys as _sys
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    if str(here) not in _sys.path:
        _sys.path.insert(0, str(here))
    import RMT  # type: ignore
    import torch as _torch

    if phi is None:
        phi = _torch.tanh

    alpha = wl._validate_alpha(alpha)
    q_grid = np.asarray(q_grid, dtype=float)
    sv_grid = np.asarray(sv_grid, dtype=float)

    if q_star is None:
        q_star = RMT.q_star_MC(alpha, sigma_w, sigma_b=sigma_b, phi=phi,
                               seed=seed)[-1]

    if seed is not None:
        _torch.manual_seed(int(seed))
    sing_vals_t = _torch.tensor(sv_grid)
    pop_size = 1
    g1 = g2 = None
    phi_prime = _torch.func.vmap(_torch.func.vmap(_torch.func.grad(phi)))
    for i in range(int(num_doublings)):
        pop_size *= 2
        if alpha != 2.0:
            stable_samples = RMT.stable_dist_sample(
                alpha, scale=2.0 ** (-1.0 / alpha), size=(num_chis, pop_size),
            )
        else:
            stable_samples = _torch.randn(num_chis, pop_size)
        chi_samples = sigma_w * phi_prime(q_star ** (1.0 / alpha) * stable_samples)
        g1, g2 = RMT.cavity_svd_resolvent(
            sing_vals_t, alpha, chi_samples,
            num_steps=pop_size ** 2,
            g1=None if i == 0 else g1.repeat(1, 1, 2),
            g2=None if i == 0 else g2.repeat(1, 1, 2),
            progress=False,
        )

    # g1, g2: shape (len(sv), num_chis, pop_size), complex.
    # rho(lambda) from the bipartite resolvent (cf. RMT.resolvent_pdf):
    #   rho = (Im g1.sum + Im g2.sum) / (pi * (pop_size_r + pop_size_c))
    g1_np = g1.cpu().numpy()
    g2_np = g2.cpu().numpy()
    # Cavity resolvent for SV problem has Im g > 0 (from -1/(lambda + ...) form
    # in cavity_svd_resolvent); the spectral density per RMT.resolvent_pdf uses
    # Im directly (no negation).
    im_g1 = np.maximum(0.0, g1_np.imag).reshape(g1_np.shape[0], -1)
    im_g2 = np.maximum(0.0, g2_np.imag).reshape(g2_np.shape[0], -1)
    rho = (im_g1.mean(axis=-1) + im_g2.mean(axis=-1)) / np.pi  # shape (len(sv),)

    log_N = float(np.log(max(int(N_reference), 2)))

    def _Dq_from_pool(im_g_pool: np.ndarray) -> np.ndarray:
        """Compute D_q at each sv, q from one side's pool."""
        # im_g_pool: shape (len(sv), pool_total)
        Dq = np.full((im_g_pool.shape[0], q_grid.size), np.nan, dtype=float)
        for j, q in enumerate(q_grid):
            if abs(float(q) - 1.0) < 1e-12:
                Dq[:, j] = 1.0
                continue
            M_q = np.mean(im_g_pool ** float(q), axis=-1)
            denom = (np.pi * np.maximum(rho, 1e-15)) ** float(q)
            ratio = M_q / np.maximum(denom, 1e-300)
            ok = (ratio > 0) & np.isfinite(ratio)
            Dq[ok, j] = 1.0 - np.log(ratio[ok]) / ((float(q) - 1.0) * log_N)
        return Dq

    Dq_left = _Dq_from_pool(im_g1)      # row side -> left SVs
    Dq_right = _Dq_from_pool(im_g2)     # column side -> right SVs

    return {
        "sv_grid": sv_grid,
        "q_grid": q_grid,
        "rho": rho,
        "Dq_left": Dq_left,         # shape (len(sv), len(q))
        "Dq_right": Dq_right,
        "N_reference": int(N_reference),
        "pool_size": int(pop_size),
        "num_chis": int(num_chis),
        "q_star": float(q_star),
    }

