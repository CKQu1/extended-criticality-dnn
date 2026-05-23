"""Heavy-tailed Wishart / singular-value density following Belinschi et al. (2008).

This module implements the limiting singular-value density of rectangular
heavy-tailed random matrices through the covariance-matrix characterization of
Belinschi, Dembo, and Guionnet.  If X is an N x M matrix with iid entries in the
domain of attraction of an alpha-stable law, the paper studies the limiting
spectral measure of

    W = a_{N+M}^{-2} X X^T,

whose eigenvalues are the squared singular values of a_{N+M}^{-1} X.

Two output normalizations are supported:

* normalization="belinschi": theory and empirics are expressed in the paper's
  quantile normalization a_{N+M};
* normalization="stable": entries are sampled directly from
  scipy.stats.levy_stable(..., scale=entry_scale) and divided by
  (N + M) ** (1 / alpha); the theory curves are rescaled accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy import optimize, special, stats


@dataclass
class WishartTheoryCurve:
    alpha: float
    gamma: float
    normalization: str
    entry_scale: float
    imag_eps: float
    quadrature_order: int
    singular_values: np.ndarray
    singular_density: np.ndarray
    squared_singular_values: np.ndarray
    squared_density: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    atom_at_zero: float


@dataclass
class EmpiricalSingularSpectrum:
    alpha: float
    gamma: float
    n_rows: int
    n_cols: int
    normalization: str
    entry_scale: float
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


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 2.0):
        raise ValueError("alpha must satisfy 0 < alpha < 2.")
    return alpha


def _validate_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must satisfy 0 < gamma <= 1.")
    return gamma


def _validate_normalization(normalization: str) -> str:
    if normalization not in {"belinschi", "stable"}:
        raise ValueError("normalization must be 'belinschi' or 'stable'.")
    return normalization


def belinschi_quantile_scale(alpha: float, entry_scale: float = 1.0) -> float:
    """Asymptotic a_N / N^{1/alpha} factor for SciPy's symmetric stable law."""

    alpha = _validate_alpha(alpha)
    entry_scale = float(entry_scale)
    tail_amplitude = (2.0 / np.pi) * special.gamma(alpha) * np.sin(np.pi * alpha / 2.0)
    return float(entry_scale * tail_amplitude ** (1.0 / alpha))


def _output_scale(alpha: float, entry_scale: float, normalization: str) -> float:
    normalization = _validate_normalization(normalization)
    if normalization == "belinschi":
        return 1.0
    return belinschi_quantile_scale(alpha, entry_scale=entry_scale)


def squared_singular_tail_prefactor(
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> float:
    """Prefactor A in rho(t) ~ A * t^{-(1 + alpha / 2)}."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    scale = _output_scale(alpha, entry_scale, normalization)
    return float(alpha * gamma / (2.0 * (1.0 + gamma)) * scale ** alpha)


def singular_tail_prefactor(
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> float:
    """Prefactor B in f(s) ~ B * s^{-(1 + alpha)}."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    scale = _output_scale(alpha, entry_scale, normalization)
    return float(alpha * gamma / (1.0 + gamma) * scale ** alpha)


def singular_survival_tail_prefactor(
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> float:
    """Prefactor C in P(S > s) ~ C * s^{-alpha}."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    scale = _output_scale(alpha, entry_scale, normalization)
    return float(gamma / (1.0 + gamma) * scale ** alpha)


def asymptotic_squared_singular_density(
    t_grid: np.ndarray,
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> np.ndarray:
    """Large-t asymptotic density of squared singular values."""

    amplitude = squared_singular_tail_prefactor(
        alpha, gamma, entry_scale=entry_scale, normalization=normalization,
    )
    t = np.asarray(t_grid, dtype=float)
    density = np.zeros_like(t)
    mask = t > 0.0
    density[mask] = amplitude / t[mask] ** (1.0 + alpha / 2.0)
    return density


def asymptotic_singular_density(
    s_grid: np.ndarray,
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> np.ndarray:
    """Large-s asymptotic singular-value density."""

    amplitude = singular_tail_prefactor(
        alpha, gamma, entry_scale=entry_scale, normalization=normalization,
    )
    s = np.asarray(s_grid, dtype=float)
    density = np.zeros_like(s)
    mask = s > 0.0
    density[mask] = amplitude / s[mask] ** (1.0 + alpha)
    return density


def asymptotic_singular_survival(
    s_grid: np.ndarray,
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> np.ndarray:
    """Large-s asymptotic survival function of singular values."""

    amplitude = singular_survival_tail_prefactor(
        alpha, gamma, entry_scale=entry_scale, normalization=normalization,
    )
    s = np.asarray(s_grid, dtype=float)
    survival = np.zeros_like(s)
    mask = s > 0.0
    survival[mask] = amplitude / s[mask] ** alpha
    survival[~mask] = 1.0
    return survival


def asymptotic_singular_quantile(
    quantile: np.ndarray | float,
    alpha: float,
    gamma: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
) -> np.ndarray:
    """Extreme upper quantile of the singular-value law.

    This returns the leading asymptotic for Q(q) as q -> 1^-.
    """

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    q = np.asarray(quantile, dtype=float)
    if np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("quantile values must satisfy 0 < q < 1.")
    p = 1.0 - q
    amplitude = singular_survival_tail_prefactor(
        alpha, gamma, entry_scale=entry_scale, normalization=normalization,
    )
    return (amplitude / p) ** (1.0 / alpha)


@lru_cache(maxsize=None)
def _laguerre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("quadrature_order must be at least 8.")
    return np.polynomial.laguerre.laggauss(order)


def g_alpha(y: complex, alpha: float, *, quadrature_order: int = 96) -> complex:
    alpha = _validate_alpha(alpha)
    nodes, weights = _laguerre_rule(int(quadrature_order))
    power = nodes ** (alpha / 2.0)
    integrand = nodes ** (alpha / 2.0 - 1.0) * np.exp(-power * y)
    return complex(np.sum(weights * integrand))


def h_alpha(y: complex, alpha: float, *, quadrature_order: int = 96) -> complex:
    alpha = _validate_alpha(alpha)
    nodes, weights = _laguerre_rule(int(quadrature_order))
    power = nodes ** (alpha / 2.0)
    integrand = np.exp(-power * y)
    return complex(np.sum(weights * integrand))


def belinschi_constant(alpha: float) -> complex:
    alpha = _validate_alpha(alpha)
    return complex(
        np.exp(0.5j * np.pi * alpha)
        * special.gamma(1.0 - alpha / 2.0)
        / special.gamma(alpha / 2.0)
    )


def _asymptotic_y_pair(alpha: float, gamma: float, z: complex) -> tuple[complex, complex]:
    c_alpha = belinschi_constant(alpha)
    g0 = special.gamma(alpha / 2.0)
    z_alpha = z ** alpha
    y1 = gamma / (1.0 + gamma) * c_alpha * g0 / z_alpha
    y2 = 1.0 / (1.0 + gamma) * c_alpha * g0 / z_alpha
    return complex(y1), complex(y2)


def solve_y_pair(
    alpha: float,
    gamma: float,
    z: complex,
    *,
    quadrature_order: int = 96,
    initial_guess: Optional[tuple[complex, complex]] = None,
    solver_tol: float = 1e-10,
    solver_maxfev: int = 400,
) -> tuple[complex, complex]:
    """Solve the Belinschi coupled fixed-point equations at a single complex z."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    c_alpha = belinschi_constant(alpha)
    z_alpha = z ** alpha
    pref1 = gamma / (1.0 + gamma) * c_alpha
    pref2 = 1.0 / (1.0 + gamma) * c_alpha

    if initial_guess is None:
        initial_guess = _asymptotic_y_pair(alpha, gamma, z)

    x0 = np.array(
        [
            initial_guess[0].real,
            initial_guess[0].imag,
            initial_guess[1].real,
            initial_guess[1].imag,
        ],
        dtype=float,
    )

    def residual(values: np.ndarray) -> np.ndarray:
        y1 = values[0] + 1j * values[1]
        y2 = values[2] + 1j * values[3]
        eq1 = z_alpha * y1 - pref1 * g_alpha(y2, alpha, quadrature_order=quadrature_order)
        eq2 = z_alpha * y2 - pref2 * g_alpha(y1, alpha, quadrature_order=quadrature_order)
        return np.array([eq1.real, eq1.imag, eq2.real, eq2.imag], dtype=float)

    result = optimize.root(residual, x0, method="hybr", tol=solver_tol, options={"maxfev": solver_maxfev})
    if not result.success:
        alt_guess = _asymptotic_y_pair(alpha, gamma, z)
        alt_x0 = np.array([alt_guess[0].real, alt_guess[0].imag, alt_guess[1].real, alt_guess[1].imag])
        result = optimize.root(
            residual, alt_x0, method="hybr", tol=solver_tol, options={"maxfev": solver_maxfev},
        )
    if not result.success:
        raise RuntimeError(f"Could not solve Belinschi system at z={z!r}: {result.message}")

    y1 = complex(result.x[0], result.x[1])
    y2 = complex(result.x[2], result.x[3])
    return y1, y2


def theoretical_singular_value_curve(
    alpha: float,
    gamma: float,
    *,
    s_max: float = 8.0,
    num_points: int = 161,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    solver_tol: float = 1e-10,
    solver_maxfev: int = 400,
) -> WishartTheoryCurve:
    """Belinschi singular-value density curve on s > 0."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    if s_max <= 0.0:
        raise ValueError("s_max must be positive.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    if imag_eps <= 0.0:
        raise ValueError("imag_eps must be positive.")

    scale = _output_scale(alpha, entry_scale, normalization)
    singular_values = np.linspace(0.0, float(s_max), int(num_points))
    squared_singular_values = singular_values ** 2
    singular_density = np.zeros_like(singular_values)
    squared_density = np.full_like(singular_values, np.nan)
    y1_values = np.full(singular_values.shape, np.nan + 1j * np.nan, dtype=complex)
    y2_values = np.full(singular_values.shape, np.nan + 1j * np.nan, dtype=complex)

    guess: Optional[tuple[complex, complex]] = None
    for idx in range(num_points - 1, 0, -1):
        s_out = singular_values[idx]
        s_base = s_out / scale
        z = complex(s_base, imag_eps)
        y1, y2 = solve_y_pair(
            alpha,
            gamma,
            z,
            quadrature_order=quadrature_order,
            initial_guess=guess,
            solver_tol=solver_tol,
            solver_maxfev=solver_maxfev,
        )
        h_val = h_alpha(y1, alpha, quadrature_order=quadrature_order)
        sq_density_base = max(0.0, -h_val.imag / (np.pi * s_base ** 2))
        sv_density_base = 2.0 * s_base * sq_density_base
        squared_density[idx] = sq_density_base / scale ** 2
        singular_density[idx] = sv_density_base / scale
        y1_values[idx] = y1
        y2_values[idx] = y2
        guess = (y1, y2)

    singular_density[0] = 0.0
    return WishartTheoryCurve(
        alpha=float(alpha),
        gamma=float(gamma),
        normalization=normalization,
        entry_scale=float(entry_scale),
        imag_eps=float(imag_eps),
        quadrature_order=int(quadrature_order),
        singular_values=singular_values,
        singular_density=singular_density,
        squared_singular_values=squared_singular_values,
        squared_density=squared_density,
        y1=y1_values,
        y2=y2_values,
        atom_at_zero=float(1.0 - gamma),
    )


def sample_rectangular_levy_matrix(
    n_rows: int,
    n_cols: int,
    alpha: float,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    random_state: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a rectangular heavy-tailed matrix in the chosen normalization."""

    alpha = _validate_alpha(alpha)
    normalization = _validate_normalization(normalization)
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be positive.")

    rng = random_state if random_state is not None else np.random.default_rng()
    matrix = stats.levy_stable.rvs(
        alpha,
        0.0,
        loc=0.0,
        scale=float(entry_scale),
        size=(n_rows, n_cols),
        random_state=rng,
    )
    divisor = (n_rows + n_cols) ** (1.0 / alpha)
    if normalization == "belinschi":
        divisor *= belinschi_quantile_scale(alpha, entry_scale=entry_scale)
    return np.asarray(matrix, dtype=float) / divisor


def empirical_singular_value_spectrum(
    alpha: float,
    gamma: float,
    *,
    n_rows: int = 96,
    num_matrices: int = 16,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    bins: int = 81,
    seed: Optional[int] = None,
    singular_range: Optional[tuple[float, float]] = None,
    squared_range: Optional[tuple[float, float]] = None,
) -> EmpiricalSingularSpectrum:
    """Direct-SVD benchmark for the limiting singular-value law."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2.")
    if num_matrices < 1:
        raise ValueError("num_matrices must be at least 1.")

    n_cols = max(1, int(round(gamma * n_rows)))
    gamma_eff = n_cols / n_rows
    rng = np.random.default_rng(seed)

    singular_blocks = []
    squared_blocks = []
    for _ in range(num_matrices):
        matrix = sample_rectangular_levy_matrix(
            n_rows,
            n_cols,
            alpha,
            entry_scale=entry_scale,
            normalization=normalization,
            random_state=rng,
        )
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        singular_blocks.append(singular_values)
        squared_blocks.append(singular_values ** 2)

    all_singular = np.concatenate(singular_blocks)
    all_squared = np.concatenate(squared_blocks)

    sv_density, sv_edges = np.histogram(all_singular, bins=bins, range=singular_range, density=True)
    sq_density, sq_edges = np.histogram(all_squared, bins=bins, range=squared_range, density=True)
    return EmpiricalSingularSpectrum(
        alpha=float(alpha),
        gamma=float(gamma_eff),
        n_rows=n_rows,
        n_cols=n_cols,
        normalization=normalization,
        entry_scale=float(entry_scale),
        num_matrices=num_matrices,
        seed=seed,
        singular_values=all_singular,
        squared_singular_values=all_squared,
        sv_bin_edges=sv_edges,
        sv_bin_centers=0.5 * (sv_edges[1:] + sv_edges[:-1]),
        sv_density=sv_density,
        sq_bin_edges=sq_edges,
        sq_bin_centers=0.5 * (sq_edges[1:] + sq_edges[:-1]),
        sq_density=sq_density,
    )
