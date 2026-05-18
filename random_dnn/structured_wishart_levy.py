"""Structured Wishart-Lévy law — numerical implementation of
``structured_wishart_levy.md``.

**Theorem 1 (general two-sided deterministic profile tau).** The limiting
squared-singular-value law of ``W = a_{N+M}^{-2} X X^T`` with
``X(i,j) = tau(i/N, j/M) x_{ij}`` is characterised by the coupled functional
fixed point

    z^a Y_r(x) = (gamma/(1+gamma)) C_a ∫ |tau(x,v)|^a g_a(Y_c(v)) dv
    z^a Y_c(y) = (1/(1+gamma))     C_a ∫ |tau(w,y)|^a g_a(Y_r(w)) dw

collapsing to  G_nu(zeta) = (1/zeta) <h_a(Y_r(., sqrt zeta))>  (atom 1-gamma
at 0).  This module discretises the (x, y) integrals on Gauss-Legendre grids
and solves the coupled system by damped Gauss-Seidel, seeded from the
``wishart_levy`` two-scalar physical-branch anchor with high->low continuation.

**Theorem 2 (one-sided tau(x,y) = c(y)).** The row field collapses to a single
scalar; that case is the validated prototype ``column_scaled_wishart_levy.py``
and is used here only as a validation gate (it is *not* re-implemented or
modified here).

**Shared-rule h (quadrature-consistency requirement of the derivation).**
``h_a := 1 - (a/2) y g_a(y)`` is evaluated from the *same* Gauss-Laguerre rule
as ``g_a``, so the collapse identity is exact in quadrature.  (The legacy
``wishart_levy.h_alpha`` / ``column_scaled`` density use an independent
Laguerre rule and therefore differ by ~5e-3 at order 64-128; field-level
checks are exact, density-level checks carry that documented gap unless both
sides use the shared-rule h.)

Scope: per the derivation, no cdf / quantile machinery; the tail prefactor and
asymptotic density (Theorem 1(v)) are in scope.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import numpy as np
from scipy import optimize, stats

import wishart_levy as wl

# tau spec: a float (constant), an (x, y) -> value callable broadcastable over
# 1-D arrays, or a 2-D array resampled onto the quadrature nodes.
ProfileSpec = Union[float, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]

_EXP_CLIP = 700.0  # see column_scaled_wishart_levy.py: keep exp() finite so a
# bad Gauss-Seidel/df-sane probe yields a large finite residual, not NaN.


@dataclass
class StructuredTheoryCurve:
    alpha: float
    gamma: float
    normalization: str
    entry_scale: float
    profile_name: str
    imag_eps: float
    quadrature_order: int
    profile_order: int
    singular_values: np.ndarray
    singular_density: np.ndarray
    squared_singular_values: np.ndarray
    squared_density: np.ndarray
    row_nodes: np.ndarray
    y_row: np.ndarray            # shape (num_points, profile_order)
    profile_alpha_moment: float  # ∫∫ |tau|^a dx dy
    atom_at_zero: float


@dataclass
class EmpiricalStructuredSpectrum:
    alpha: float
    gamma: float
    n_rows: int
    n_cols: int
    normalization: str
    entry_scale: float
    profile_name: str
    num_matrices: int
    seed: Optional[int]
    singular_values: np.ndarray
    squared_singular_values: np.ndarray
    sv_bin_edges: np.ndarray
    sv_bin_centers: np.ndarray
    sv_density: np.ndarray
    sq_bin_edges: np.ndarray
    sq_bin_centers: np.ndarray
    sq_density: np.ndarray


# --- shared alpha-stable engine (Laguerre rule reused from wishart_levy) -----


@lru_cache(maxsize=None)
def _legendre01(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights mapped to [0, 1] (weights sum to 1)."""
    if order < 8:
        raise ValueError("profile_order must be at least 8.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _g_alpha_vec(values: np.ndarray, alpha: float, quadrature_order: int) -> np.ndarray:
    """Overflow-safe vectorised g_alpha on an array of complex arguments."""
    arr = np.atleast_1d(np.asarray(values, dtype=complex))
    nodes, lw = wl._laguerre_rule(int(quadrature_order))
    powers = nodes ** (alpha / 2.0)
    prefactor = lw * nodes ** (alpha / 2.0 - 1.0)
    expo = -powers[:, None] * arr[None, :]
    expo = np.clip(expo.real, -_EXP_CLIP, _EXP_CLIP) + 1j * expo.imag
    return np.sum(prefactor[:, None] * np.exp(expo), axis=0)


def _h_alpha_vec(values: np.ndarray, alpha: float, quadrature_order: int) -> np.ndarray:
    """Shared-rule h := 1 - (alpha/2) y g_alpha(y), same Laguerre nodes as g."""
    arr = np.atleast_1d(np.asarray(values, dtype=complex))
    return 1.0 - (alpha / 2.0) * arr * _g_alpha_vec(arr, alpha, quadrature_order)


# --- profile machinery -----------------------------------------------------


def _profile_alpha_matrix(
    tau: ProfileSpec,
    x_nodes: np.ndarray,
    y_nodes: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, str]:
    """Return (|tau(x_i, y_j)|^alpha matrix of shape (n_x, n_y), name)."""
    if isinstance(tau, (int, float)):
        T = np.full((x_nodes.size, y_nodes.size), float(tau), dtype=float)
        name = f"constant({float(tau):.3g})"
    elif callable(tau):
        XX, YY = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        T = np.asarray(tau(XX, YY), dtype=float)
        if T.shape != (x_nodes.size, y_nodes.size):
            raise ValueError("callable tau must broadcast to (n_x, n_y).")
        name = "callable"
    elif isinstance(tau, np.ndarray):
        a = np.asarray(tau, dtype=float)
        if a.ndim != 2:
            raise ValueError("array tau must be 2-D.")
        gx = np.linspace(0.0, 1.0, a.shape[0])
        gy = np.linspace(0.0, 1.0, a.shape[1])
        # bilinear resample onto the quadrature nodes
        ix = np.interp(x_nodes, gx, np.arange(a.shape[0]))
        iy = np.interp(y_nodes, gy, np.arange(a.shape[1]))
        i0 = np.clip(np.floor(ix).astype(int), 0, a.shape[0] - 2)
        j0 = np.clip(np.floor(iy).astype(int), 0, a.shape[1] - 2)
        fx = (ix - i0)[:, None]
        fy = (iy - j0)[None, :]
        T = (
            a[i0][:, j0] * (1 - fx) * (1 - fy)
            + a[i0 + 1][:, j0] * fx * (1 - fy)
            + a[i0][:, j0 + 1] * (1 - fx) * fy
            + a[i0 + 1][:, j0 + 1] * fx * fy
        )
        name = f"array[{a.shape[0]}x{a.shape[1]}]"
    else:
        raise ValueError("tau must be a float, a callable, or a 2-D array.")
    if np.any(~np.isfinite(T)) or np.any(T < 0.0):
        raise ValueError("tau must be finite and non-negative.")
    return np.abs(T) ** alpha, name


def profile_alpha_moment(
    tau: ProfileSpec, alpha: float, *, profile_order: int = 96
) -> float:
    """∫∫_{[0,1]^2} |tau|^alpha dx dy via Gauss-Legendre."""
    alpha = wl._validate_alpha(alpha)
    xn, xw = _legendre01(int(profile_order))
    yn, yw = _legendre01(int(profile_order))
    T, _ = _profile_alpha_matrix(tau, xn, yn, alpha)
    return float(xw @ T @ yw)


# --- coupled two-field solver (Theorem 1) ----------------------------------


def _coupled_rhs(
    Y_r: np.ndarray,
    Y_c: np.ndarray,
    *,
    z_alpha: complex,
    alpha: float,
    gamma: float,
    c_alpha: complex,
    T: np.ndarray,
    x_w: np.ndarray,
    y_w: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One application of the Theorem 1(ii) maps (Gauss-Seidel order)."""
    pref_r = gamma * c_alpha / ((1.0 + gamma) * z_alpha)
    pref_c = c_alpha / ((1.0 + gamma) * z_alpha)
    g_r = _g_alpha_vec(Y_r, alpha, quadrature_order)
    Y_c_new = pref_c * (T.T @ (x_w * g_r))
    g_c = _g_alpha_vec(Y_c_new, alpha, quadrature_order)
    Y_r_new = pref_r * (T @ (y_w * g_c))
    return Y_r_new, Y_c_new


def _coupled_residual(
    Y_r: np.ndarray,
    Y_c: np.ndarray,
    *,
    z_alpha: complex,
    alpha: float,
    gamma: float,
    c_alpha: complex,
    T: np.ndarray,
    x_w: np.ndarray,
    y_w: np.ndarray,
    quadrature_order: int,
) -> float:
    """sup-norm of the Theorem 1(ii) coupled residual at (Y_r, Y_c)."""
    g_r = _g_alpha_vec(Y_r, alpha, quadrature_order)
    g_c = _g_alpha_vec(Y_c, alpha, quadrature_order)
    res_r = z_alpha * Y_r - (gamma * c_alpha / (1.0 + gamma)) * (T @ (y_w * g_c))
    res_c = z_alpha * Y_c - (c_alpha / (1.0 + gamma)) * (T.T @ (x_w * g_r))
    if not (np.all(np.isfinite(res_r)) and np.all(np.isfinite(res_c))):
        return np.inf
    return float(max(np.max(np.abs(res_r)), np.max(np.abs(res_c))))


def solve_structured_fields(
    alpha: float,
    gamma: float,
    z: complex,
    *,
    T: np.ndarray,
    x_nodes: np.ndarray,
    x_w: np.ndarray,
    y_nodes: np.ndarray,
    y_w: np.ndarray,
    quadrature_order: int = 96,
    seed: Optional[tuple[np.ndarray, np.ndarray]] = None,
    wish_seed: Optional[tuple[complex, complex]] = None,
    tol: float = 1e-12,
    max_iter: int = 2000,
    _homotopy: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple[complex, complex]]:
    """Solve the coupled (Y_r, Y_c) fixed point at a single complex z.

    Returns ``(Y_r, Y_c, (y1, y2))`` where ``(y1, y2)`` is the wishart_levy
    two-scalar anchor used (to be threaded back as ``wish_seed`` for
    continuation, exactly as in column_scaled_wishart_levy.py).  Damped
    Gauss-Seidel from the anchor / carried seed, df-sane fallback.
    """
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    c_alpha = wl.belinschi_constant(alpha)
    z_alpha = z ** alpha
    n_x, n_y = T.shape
    kw = dict(z_alpha=z_alpha, alpha=alpha, gamma=gamma, c_alpha=c_alpha,
              T=T, x_w=x_w, y_w=y_w, quadrature_order=quadrature_order)
    accept = max(tol, 1e-9)

    # Physical-branch anchor at *this* z: the wishart_levy two-scalar solution
    # broadcast across the grids.  Exact for tau == const (Corollary 1); the
    # correct branch otherwise.  Recomputed every z (not continuation-only),
    # mirroring the proven column_scaled_wishart_levy.py robustness pattern.
    try:
        y1, y2 = wl.solve_y_pair(alpha, gamma, z, quadrature_order=quadrature_order,
                                 initial_guess=wish_seed)
    except RuntimeError:
        g0 = wl.g_alpha(0.0, alpha, quadrature_order=quadrature_order)
        y1 = gamma / (1.0 + gamma) * c_alpha * g0 / z_alpha
        y2 = 1.0 / (1.0 + gamma) * c_alpha * g0 / z_alpha
    wish_pair = (complex(y1), complex(y2))
    anchor = (np.full(n_x, complex(y1)), np.full(n_y, complex(y2)))

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    if seed is not None:
        candidates.append((np.asarray(seed[0], dtype=complex),
                           np.asarray(seed[1], dtype=complex)))
    candidates.append(anchor)

    def _pack(Y_r: np.ndarray, Y_c: np.ndarray) -> np.ndarray:
        return np.concatenate([Y_r.real, Y_r.imag, Y_c.real, Y_c.imag])

    def _unpack(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (v[:n_x] + 1j * v[n_x:2 * n_x],
                v[2 * n_x:2 * n_x + n_y] + 1j * v[2 * n_x + n_y:])

    def residual(v: np.ndarray) -> np.ndarray:
        yr, yc = _unpack(v)
        g_r = _g_alpha_vec(yr, alpha, quadrature_order)
        g_c = _g_alpha_vec(yc, alpha, quadrature_order)
        res_r = z_alpha * yr - (gamma * c_alpha / (1.0 + gamma)) * (T @ (y_w * g_c))
        res_c = z_alpha * yc - (c_alpha / (1.0 + gamma)) * (T.T @ (x_w * g_r))
        out = np.concatenate([res_r.real, res_r.imag, res_c.real, res_c.imag])
        return np.where(np.isfinite(out), out, 1e6)

    maxfev = 300 * (2 * (n_x + n_y) + 1)
    best: Optional[tuple[np.ndarray, np.ndarray]] = None
    best_r = np.inf
    for Y_r0, Y_c0 in candidates:
        if not (np.all(np.isfinite(Y_r0)) and np.all(np.isfinite(Y_c0))):
            continue
        if _coupled_residual(Y_r0, Y_c0, **kw) < accept:  # seed solves it
            return Y_r0, Y_c0, wish_pair
        # cheap damped Gauss-Seidel warm-up to improve the hybr seed
        Y_r, Y_c = Y_r0.copy(), Y_c0.copy()
        theta = 1.0
        for _ in range(120):
            Y_r_new, Y_c_new = _coupled_rhs(Y_r, Y_c, **kw)
            if not (np.all(np.isfinite(Y_r_new)) and np.all(np.isfinite(Y_c_new))):
                theta *= 0.5
                if theta < 1e-3:
                    break
                continue
            Y_r = (1.0 - theta) * Y_r + theta * Y_r_new
            Y_c = (1.0 - theta) * Y_c + theta * Y_c_new
            if _coupled_residual(Y_r, Y_c, **kw) < accept:
                return Y_r, Y_c, wish_pair
        # Newton-type solve (proven-robust path, as in column_scaled hybr)
        for x0 in (_pack(Y_r, Y_c), _pack(Y_r0, Y_c0)):
            sol = optimize.root(residual, x0, method="hybr",
                                options={"maxfev": maxfev})
            yr, yc = _unpack(sol.x)
            r = _coupled_residual(yr, yc, **kw)
            if sol.success and r < accept:
                return yr, yc, wish_pair
            if r < best_r:
                best_r, best = r, (yr, yc)

    # --- imag-eps homotopy: solve on a broadened (smoother) contour and
    #     sharpen back, keeping the physical branch at stiff (small-z,
    #     gamma<1, alpha>~1.5) points where a direct hybr finds a spurious
    #     root.  Recursion guarded by _homotopy.
    if _homotopy and z.imag > 0:
        hseed = None
        chain_ok = True
        for fac in (8.0, 4.0, 2.0, 1.0):
            zz = complex(z.real, z.imag * fac)
            try:
                Yr_h, Yc_h, _ = solve_structured_fields(
                    alpha, gamma, zz, T=T, x_nodes=x_nodes, x_w=x_w,
                    y_nodes=y_nodes, y_w=y_w,
                    quadrature_order=quadrature_order, seed=hseed,
                    wish_seed=wish_seed, tol=tol, max_iter=max_iter,
                    _homotopy=False,
                )
            except RuntimeError:
                chain_ok = False
                break
            hseed = (Yr_h, Yc_h)
        if chain_ok and hseed is not None:
            if _coupled_residual(hseed[0], hseed[1], **kw) < accept:
                return hseed[0], hseed[1], wish_pair

    if best is not None and best_r < 1e-6:
        return best[0], best[1], wish_pair     # usable bulk accuracy
    raise RuntimeError(
        f"structured fields did not converge at z={z!r} "
        f"(best residual {best_r:.2e})"
    )


# --- theory curve (Theorem 1: collapse -> density + atom) ------------------


def theoretical_structured_singular_value_curve(
    alpha: float,
    gamma: float,
    tau: ProfileSpec,
    *,
    s_max: float = 8.0,
    num_points: int = 161,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    profile_order: int = 32,
    tol: float = 1e-12,
) -> StructuredTheoryCurve:
    """Singular-value density of the structured law via the Theorem 1 collapse
    ``G_nu = (1/zeta) <h_alpha(Y_r(., sqrt zeta))>`` (shared-rule h)."""
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    normalization = wl._validate_normalization(normalization)
    if s_max <= 0.0 or num_points < 3 or imag_eps <= 0.0:
        raise ValueError("need s_max>0, num_points>=3, imag_eps>0.")

    x_nodes, x_w = _legendre01(int(profile_order))
    y_nodes, y_w = _legendre01(int(profile_order))
    T, profile_name = _profile_alpha_matrix(tau, x_nodes, y_nodes, alpha)
    moment = float(x_w @ T @ y_w)
    output_scale = wl._output_scale(alpha, entry_scale, normalization)

    singular_values = np.linspace(0.0, float(s_max), int(num_points))
    singular_density = np.zeros_like(singular_values)
    squared_density = np.full_like(singular_values, np.nan)
    y_row = np.full((int(num_points), int(profile_order)),
                    np.nan + 1j * np.nan, dtype=complex)

    seed: Optional[tuple[np.ndarray, np.ndarray]] = None
    wish_seed: Optional[tuple[complex, complex]] = None
    for idx in range(int(num_points) - 1, 0, -1):
        s_out = singular_values[idx]
        s_base = s_out / output_scale
        z = complex(s_base, imag_eps)
        Y_r, Y_c, wish_seed = solve_structured_fields(
            alpha, gamma, z, T=T, x_nodes=x_nodes, x_w=x_w,
            y_nodes=y_nodes, y_w=y_w, quadrature_order=quadrature_order,
            seed=seed, wish_seed=wish_seed, tol=tol,
        )
        seed = (Y_r, Y_c)
        H_avg = complex(x_w @ _h_alpha_vec(Y_r, alpha, quadrature_order))
        sv_density_base = max(0.0, -2.0 * H_avg.imag / (np.pi * s_base))
        singular_density[idx] = sv_density_base / output_scale
        squared_density[idx] = singular_density[idx] / (2.0 * s_out)
        y_row[idx] = Y_r

    singular_density[0] = 0.0
    return StructuredTheoryCurve(
        alpha=float(alpha), gamma=float(gamma), normalization=normalization,
        entry_scale=float(entry_scale), profile_name=profile_name,
        imag_eps=float(imag_eps), quadrature_order=int(quadrature_order),
        profile_order=int(profile_order), singular_values=singular_values,
        singular_density=singular_density,
        squared_singular_values=singular_values ** 2,
        squared_density=squared_density, row_nodes=x_nodes, y_row=y_row,
        profile_alpha_moment=moment, atom_at_zero=float(1.0 - gamma),
    )


# --- tail (Theorem 1(v)) ---------------------------------------------------


def structured_singular_tail_prefactor(
    alpha: float,
    gamma: float,
    tau: ProfileSpec,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    profile_order: int = 96,
    atomless: bool = False,
) -> float:
    """Prefactor B in the singular-value tail  f(s) ~ B s^{-1-alpha}.

    Theorem 1(v): t^{1+a/2} rho_nu(t) -> (a*gamma/(2(1+gamma))) ∫∫|tau|^a.
    ``atomless`` selects mu_SV (drop the gamma) vs the Gram mu_SV^{(N)}.
    """
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    K = profile_alpha_moment(tau, alpha, profile_order=profile_order)
    scale = wl._output_scale(alpha, entry_scale, normalization)
    g_factor = 1.0 if atomless else gamma
    return float(alpha * g_factor / (1.0 + gamma) * K * scale ** alpha)


def asymptotic_singular_density(
    s_grid: np.ndarray,
    alpha: float,
    gamma: float,
    tau: ProfileSpec,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    profile_order: int = 96,
    atomless: bool = False,
) -> np.ndarray:
    amp = structured_singular_tail_prefactor(
        alpha, gamma, tau, entry_scale=entry_scale,
        normalization=normalization, profile_order=profile_order,
        atomless=atomless,
    )
    s = np.asarray(s_grid, dtype=float)
    out = np.zeros_like(s)
    m = s > 0.0
    out[m] = amp / s[m] ** (1.0 + alpha)
    return out


# --- empirical structured SVD (validation gate 3) --------------------------


def sample_structured_levy_matrix(
    n_rows: int,
    alpha: float,
    gamma: float,
    tau: ProfileSpec,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    random_state: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, str]:
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    normalization = wl._validate_normalization(normalization)
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2.")
    n_cols = max(1, int(round(gamma * n_rows)))
    rng = random_state if random_state is not None else np.random.default_rng()
    x = (np.arange(n_rows) + 0.5) / n_rows
    y = (np.arange(n_cols) + 0.5) / n_cols
    if isinstance(tau, (int, float)):
        Tprofile = np.full((n_rows, n_cols), float(tau))
        name = f"constant({float(tau):.3g})"
    elif callable(tau):
        XX, YY = np.meshgrid(x, y, indexing="ij")
        Tprofile = np.asarray(tau(XX, YY), dtype=float)
        name = "callable"
    else:
        Tprofile, name = _profile_alpha_matrix(tau, x, y, 1.0)  # |tau| via ^1
    M = stats.levy_stable.rvs(alpha, 0.0, loc=0.0, scale=float(entry_scale),
                              size=(n_rows, n_cols), random_state=rng)
    M = np.asarray(M, dtype=float) * Tprofile
    divisor = (n_rows + n_cols) ** (1.0 / alpha)
    if normalization == "belinschi":
        divisor *= wl.belinschi_quantile_scale(alpha, entry_scale=entry_scale)
    return M / divisor, name


def empirical_structured_singular_value_spectrum(
    alpha: float,
    gamma: float,
    tau: ProfileSpec,
    *,
    n_rows: int = 400,
    num_matrices: int = 40,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    bins: int = 121,
    seed: Optional[int] = None,
    singular_range: Optional[tuple[float, float]] = None,
    squared_range: Optional[tuple[float, float]] = None,
) -> EmpiricalStructuredSpectrum:
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    rng = np.random.default_rng(seed)
    n_cols = max(1, int(round(gamma * n_rows)))
    sv, sq, name = [], [], None
    for _ in range(num_matrices):
        M, name = sample_structured_levy_matrix(
            n_rows, alpha, gamma, tau, entry_scale=entry_scale,
            normalization=normalization, random_state=rng,
        )
        s = np.linalg.svd(M, compute_uv=False)
        sv.append(s)
        sq.append(s ** 2)
    sv = np.concatenate(sv)
    sq = np.concatenate(sq)
    svd, sve = np.histogram(sv, bins=bins, range=singular_range, density=True)
    sqd, sqe = np.histogram(sq, bins=bins, range=squared_range, density=True)
    return EmpiricalStructuredSpectrum(
        alpha=float(alpha), gamma=float(n_cols / n_rows), n_rows=n_rows,
        n_cols=n_cols, normalization=normalization,
        entry_scale=float(entry_scale), profile_name=str(name),
        num_matrices=num_matrices, seed=seed, singular_values=sv,
        squared_singular_values=sq, sv_bin_edges=sve,
        sv_bin_centers=0.5 * (sve[1:] + sve[:-1]), sv_density=svd,
        sq_bin_edges=sqe, sq_bin_centers=0.5 * (sqe[1:] + sqe[:-1]),
        sq_density=sqd,
    )


# --- validation gates ------------------------------------------------------


def compare_constant_to_wishart(
    alpha: float,
    gamma: float,
    *,
    s_max: float = 6.0,
    num_points: int = 81,
    imag_eps: float = 1e-2,
    quadrature_order: int = 64,
    profile_order: int = 24,
) -> dict[str, float]:
    """Gate 1: tau == 1 must reproduce ``wishart_levy`` (Corollary 1).

    Field-level: structured Y_r (constant in x) vs wishart_levy Y_1, exact.
    Density-level: compared with the *shared-rule* h recomputed on
    wishart_levy's solved Y_1 (so the documented independent-rule h gap does
    not contaminate the gate)."""
    wcur = wl.theoretical_singular_value_curve(
        alpha, gamma, s_max=s_max, num_points=num_points, imag_eps=imag_eps,
        quadrature_order=quadrature_order,
    )
    scur = theoretical_structured_singular_value_curve(
        alpha, gamma, 1.0, s_max=s_max, num_points=num_points,
        imag_eps=imag_eps, quadrature_order=quadrature_order,
        profile_order=profile_order,
    )
    scale = wl._output_scale(alpha, 1.0, "stable")
    max_field_support = 0.0   # field diff where the spectrum is non-trivial
    max_field_all = 0.0       # incl. spectrally-empty points (branch-ambiguous)
    max_dens = 0.0
    for idx in range(num_points - 1, 0, -1):
        y1 = wcur.y1[idx]
        Yr = scur.y_row[idx]
        if not np.isfinite(y1) or not np.all(np.isfinite(Yr)):
            continue
        fd = float(np.max(np.abs(Yr - y1)))
        max_field_all = max(max_field_all, fd)
        s_base = scur.singular_values[idx] / scale
        h1 = complex(_h_alpha_vec(np.array([y1]), alpha, quadrature_order)[0])
        d_w = max(0.0, -2.0 * h1.imag / (np.pi * s_base)) / scale
        max_dens = max(max_dens, abs(d_w - scur.singular_density[idx]))
        # Off the support Y is a branch-dependent analytic continuation
        # (the theorem fixes the *measure*, not Y off the cut), so the
        # meaningful field check is restricted to where the density is
        # non-negligible.
        if scur.singular_density[idx] > 1e-6:
            max_field_support = max(max_field_support, fd)
    return {
        "max_field_diff_on_support": max_field_support,
        "max_field_diff_all_pts": max_field_all,
        "max_density_diff(shared-rule)": max_dens,
        "profile_alpha_moment": scur.profile_alpha_moment,
    }


def compare_one_sided_to_column_scaled(
    alpha: float,
    gamma: float,
    c_profile: Callable[[np.ndarray], np.ndarray],
    *,
    s_max: float = 6.0,
    num_points: int = 81,
    imag_eps: float = 1e-2,
    quadrature_order: int = 64,
    profile_order: int = 96,
) -> dict[str, float]:
    """Gate 2: tau(x,y)=c(y) must reproduce the Theorem 2 prototype
    ``column_scaled_wishart_levy.py`` at the *field* level (both use the same
    shared-rule g closure -> exact).  Density differs only by the legacy
    independent-rule h gap in column_scaled, which is expected."""
    import column_scaled_wishart_levy as cwl

    scur = theoretical_structured_singular_value_curve(
        alpha, gamma, (lambda X, Y: c_profile(Y)), s_max=s_max,
        num_points=num_points, imag_eps=imag_eps,
        quadrature_order=quadrature_order, profile_order=profile_order,
    )
    ccur = cwl.theoretical_column_scaled_singular_value_curve(
        alpha, gamma, profile=c_profile, s_max=s_max, num_points=num_points,
        imag_eps=imag_eps, quadrature_order=quadrature_order,
        profile_order=profile_order,
    )
    max_field_support = 0.0   # where the spectrum is non-trivial
    max_field_all = 0.0       # incl. branch-ambiguous spectrally-empty pts
    max_dens = 0.0
    for idx in range(num_points - 1, 0, -1):
        yr_struct = scur.y_row[idx]
        yr_col = ccur.y_row[idx]
        if not np.isfinite(yr_col) or not np.all(np.isfinite(yr_struct)):
            continue
        # one-sided => Y_r constant in x; compare to the column scalar
        fd = float(np.max(np.abs(yr_struct - yr_col)))
        max_field_all = max(max_field_all, fd)
        if scur.singular_density[idx] > 1e-6:
            max_field_support = max(max_field_support, fd)
        max_dens = max(max_dens, abs(scur.singular_density[idx]
                                     - ccur.singular_density[idx]))
    return {
        "max_field_diff_on_support": max_field_support,
        "max_field_diff_all_pts": max_field_all,
        "max_density_diff(legacy-h gap expected ~5e-3)": max_dens,
        "row_field_x_spread": float(
            np.nanmax([np.ptp(np.abs(r)) for r in scur.y_row if np.all(np.isfinite(r))])
        ),
    }
