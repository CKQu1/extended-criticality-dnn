"""Column-scaled Wishart-Lévy prototype following the Belinschi profile ansatz.

This module implements the scalar closure written in
``column_scaled_wishart_levy.md`` for a rectangular heavy-tailed random matrix
whose columns carry a deterministic profile ``c(u)``.  The derivation is based
on the general profile theorem of Belinschi--Dembo--Guionnet specialized to the
Hermitization of a column-scaled rectangular matrix.

The implementation is a numerical prototype of that specialization.  It is
anchored by one mandatory sanity check: for a constant profile ``c(u) == 1`` it
should reproduce the ordinary Wishart-Lévy singular-value law implemented in
``wishart_levy.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Union

import numpy as np
from scipy import optimize, stats

import wishart_levy as wl


ProfileSpec = Union[str, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass
class ColumnScaledTheoryCurve:
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
    y_row: np.ndarray
    profile_nodes: np.ndarray
    profile_values: np.ndarray
    profile_alpha_moment: float
    atom_at_zero: float


@dataclass
class EmpiricalColumnScaledSpectrum:
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
    column_profile_grid: np.ndarray
    column_profile_values: np.ndarray


def _validate_alpha(alpha: float) -> float:
    return wl._validate_alpha(alpha)


def _validate_gamma(gamma: float) -> float:
    return wl._validate_gamma(gamma)


def _validate_normalization(normalization: str) -> str:
    return wl._validate_normalization(normalization)


@lru_cache(maxsize=None)
def _legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("profile_order must be at least 8.")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _coerce_profile_values(values: np.ndarray, *, context: str) -> np.ndarray:
    profile_values = np.asarray(values, dtype=float)
    if profile_values.ndim != 1:
        raise ValueError(f"{context} must be one-dimensional.")
    if np.any(~np.isfinite(profile_values)):
        raise ValueError(f"{context} contains non-finite values.")
    if np.any(profile_values < 0.0):
        raise ValueError(f"{context} must be non-negative.")
    return profile_values


def evaluate_column_profile(
    u_grid: np.ndarray,
    profile: ProfileSpec = "constant",
    *,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> tuple[np.ndarray, str]:
    """Evaluate a deterministic column profile on ``u_grid``.

    Supported string profiles:
    - ``"constant"``
    - ``"two_level"``
    - ``"power_law"``
    """

    u = np.asarray(u_grid, dtype=float)
    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("u_grid must lie in [0, 1].")

    if callable(profile):
        values = _coerce_profile_values(np.asarray(profile(u), dtype=float), context="profile(u_grid)")
        return values, "callable"

    if isinstance(profile, np.ndarray):
        grid = np.linspace(0.0, 1.0, profile.size)
        values = np.interp(u, grid, _coerce_profile_values(profile, context="profile array"))
        return values, f"array[{profile.size}]"

    if not isinstance(profile, str):
        raise ValueError("profile must be a string family, callable, or one-dimensional array.")

    split = float(split)
    cutoff = float(cutoff)
    exponent = float(exponent)
    if not (0.0 < split < 1.0):
        raise ValueError("split must satisfy 0 < split < 1.")
    if not (0.0 < cutoff <= 1.0):
        raise ValueError("cutoff must satisfy 0 < cutoff <= 1.")

    if profile == "constant":
        values = np.full_like(u, float(constant_value), dtype=float)
        name = f"constant({float(constant_value):.3g})"
    elif profile == "two_level":
        values = np.where(u < split, float(left_value), float(right_value))
        name = f"two_level(split={split:.3g}, left={float(left_value):.3g}, right={float(right_value):.3g})"
    elif profile == "power_law":
        values = float(constant_value) * np.maximum(u, cutoff) ** (-exponent)
        name = f"power_law(A={float(constant_value):.3g}, exponent={exponent:.3g}, cutoff={cutoff:.3g})"
    else:
        raise ValueError("unknown profile family.")

    values = _coerce_profile_values(values, context="profile values")
    return values, name


def _profile_quadrature(
    alpha: float,
    profile: ProfileSpec,
    *,
    profile_order: int,
    constant_value: float,
    left_value: float,
    right_value: float,
    split: float,
    exponent: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    nodes, weights = _legendre_rule(int(profile_order))
    values, name = evaluate_column_profile(
        nodes,
        profile,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    profile_alpha = values ** alpha
    alpha_moment = float(np.sum(weights * profile_alpha))
    return nodes, weights, values, alpha_moment, name


def _profile_columns(
    n_cols: int,
    profile: ProfileSpec,
    *,
    constant_value: float,
    left_value: float,
    right_value: float,
    split: float,
    exponent: float,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    u = (np.arange(n_cols, dtype=float) + 0.5) / n_cols
    values, name = evaluate_column_profile(
        u,
        profile,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    return u, values, name


def column_profile_alpha_moment(
    alpha: float,
    profile: ProfileSpec = "constant",
    *,
    profile_order: int = 128,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> float:
    """Return ∫_0^1 c(u)^alpha du for the chosen profile."""

    alpha = _validate_alpha(alpha)
    _, _, _, alpha_moment, _ = _profile_quadrature(
        alpha,
        profile,
        profile_order=profile_order,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    return alpha_moment


def _output_scale(alpha: float, entry_scale: float, normalization: str) -> float:
    return wl._output_scale(alpha, entry_scale, normalization)


def squared_singular_tail_prefactor(
    alpha: float,
    gamma: float,
    profile: ProfileSpec = "constant",
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    profile_order: int = 128,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> float:
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    alpha_moment = column_profile_alpha_moment(
        alpha,
        profile,
        profile_order=profile_order,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    scale = _output_scale(alpha, entry_scale, normalization)
    return float(alpha * gamma / (2.0 * (1.0 + gamma)) * alpha_moment * scale ** alpha)


def singular_tail_prefactor(
    alpha: float,
    gamma: float,
    profile: ProfileSpec = "constant",
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    profile_order: int = 128,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> float:
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    alpha_moment = column_profile_alpha_moment(
        alpha,
        profile,
        profile_order=profile_order,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    scale = _output_scale(alpha, entry_scale, normalization)
    return float(alpha * gamma / (1.0 + gamma) * alpha_moment * scale ** alpha)


def singular_survival_tail_prefactor(
    alpha: float,
    gamma: float,
    profile: ProfileSpec = "constant",
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    profile_order: int = 128,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> float:
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    alpha_moment = column_profile_alpha_moment(
        alpha,
        profile,
        profile_order=profile_order,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    scale = _output_scale(alpha, entry_scale, normalization)
    return float(gamma / (1.0 + gamma) * alpha_moment * scale ** alpha)


def asymptotic_squared_singular_density(
    t_grid: np.ndarray,
    alpha: float,
    gamma: float,
    profile: ProfileSpec = "constant",
    **kwargs: Any,
) -> np.ndarray:
    amplitude = squared_singular_tail_prefactor(alpha, gamma, profile, **kwargs)
    t = np.asarray(t_grid, dtype=float)
    density = np.zeros_like(t)
    mask = t > 0.0
    density[mask] = amplitude / t[mask] ** (1.0 + alpha / 2.0)
    return density


def asymptotic_singular_density(
    s_grid: np.ndarray,
    alpha: float,
    gamma: float,
    profile: ProfileSpec = "constant",
    **kwargs: Any,
) -> np.ndarray:
    amplitude = singular_tail_prefactor(alpha, gamma, profile, **kwargs)
    s = np.asarray(s_grid, dtype=float)
    density = np.zeros_like(s)
    mask = s > 0.0
    density[mask] = amplitude / s[mask] ** (1.0 + alpha)
    return density


def asymptotic_singular_survival(
    s_grid: np.ndarray,
    alpha: float,
    gamma: float,
    profile: ProfileSpec = "constant",
    **kwargs: Any,
) -> np.ndarray:
    amplitude = singular_survival_tail_prefactor(alpha, gamma, profile, **kwargs)
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
    profile: ProfileSpec = "constant",
    **kwargs: Any,
) -> np.ndarray:
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    q = np.asarray(quantile, dtype=float)
    if np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("quantile values must satisfy 0 < q < 1.")
    p = 1.0 - q
    amplitude = singular_survival_tail_prefactor(alpha, gamma, profile, **kwargs)
    return (amplitude / p) ** (1.0 / alpha)


def _g_alpha_vector(values: np.ndarray, alpha: float, quadrature_order: int) -> np.ndarray:
    values = np.atleast_1d(np.asarray(values, dtype=complex))
    nodes, weights = wl._laguerre_rule(int(quadrature_order))
    powers = nodes ** (alpha / 2.0)
    prefactor = weights * nodes ** (alpha / 2.0 - 1.0)
    integrand = prefactor[:, None] * np.exp(-powers[:, None] * values[None, :])
    return np.sum(integrand, axis=0)


def _h_alpha_scalar(value: complex, alpha: float, quadrature_order: int) -> complex:
    return wl.h_alpha(value, alpha, quadrature_order=quadrature_order)


def _row_residual(
    y_row: complex,
    *,
    z: complex,
    alpha: float,
    gamma: float,
    c_alpha: complex,
    profile_alpha: np.ndarray,
    profile_weights: np.ndarray,
    quadrature_order: int,
) -> complex:
    z_alpha = z ** alpha
    g_row = wl.g_alpha(y_row, alpha, quadrature_order=quadrature_order)
    y_col = (c_alpha / (1.0 + gamma)) * (profile_alpha * g_row) / z_alpha
    g_col = _g_alpha_vector(y_col, alpha, quadrature_order)
    integral = np.sum(profile_weights * profile_alpha * g_col)
    rhs = (gamma * c_alpha / (1.0 + gamma)) * integral
    return z_alpha * y_row - rhs


def _solve_row_field_with_profile(
    alpha: float,
    gamma: float,
    z: complex,
    *,
    quadrature_order: int = 96,
    profile_weights: np.ndarray,
    profile_values: np.ndarray,
    alpha_moment: float,
    initial_guess: Optional[complex] = None,
    solver_tol: float = 1e-10,
    solver_maxfev: int = 400,
) -> complex:
    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    c_alpha = wl.belinschi_constant(alpha)
    profile_alpha = profile_values ** alpha
    z_alpha = z ** alpha

    if initial_guess is None:
        g0 = wl.g_alpha(0.0, alpha, quadrature_order=quadrature_order)
        initial_guess = (gamma / (1.0 + gamma)) * c_alpha * g0 * alpha_moment / z_alpha

    x0 = np.array([initial_guess.real, initial_guess.imag], dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        y_row = complex(values[0], values[1])
        res = _row_residual(
            y_row,
            z=z,
            alpha=alpha,
            gamma=gamma,
            c_alpha=c_alpha,
            profile_alpha=profile_alpha,
            profile_weights=profile_weights,
            quadrature_order=quadrature_order,
        )
        return np.array([res.real, res.imag], dtype=float)

    result = optimize.root(residual, x0, method="hybr", tol=solver_tol, options={"maxfev": solver_maxfev})
    if not result.success:
        g0 = wl.g_alpha(0.0, alpha, quadrature_order=quadrature_order)
        alt_guess = (gamma / (1.0 + gamma)) * c_alpha * g0 * alpha_moment / z_alpha
        alt_x0 = np.array([alt_guess.real, alt_guess.imag], dtype=float)
        result = optimize.root(
            residual, alt_x0, method="hybr", tol=solver_tol, options={"maxfev": solver_maxfev},
        )
    if not result.success:
        raise RuntimeError(f"Could not solve Y_r(z) at z={z!r}: {result.message}")
    return complex(result.x[0], result.x[1])


def solve_row_field(
    alpha: float,
    gamma: float,
    z: complex,
    *,
    profile: ProfileSpec = "constant",
    quadrature_order: int = 96,
    profile_order: int = 128,
    initial_guess: Optional[complex] = None,
    solver_tol: float = 1e-10,
    solver_maxfev: int = 400,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> complex:
    """Solve the scalar Y_r(z) closure for the chosen column profile."""

    _, profile_weights, profile_values, alpha_moment, _ = _profile_quadrature(
        _validate_alpha(alpha),
        profile,
        profile_order=profile_order,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    return _solve_row_field_with_profile(
        alpha,
        gamma,
        z,
        quadrature_order=quadrature_order,
        profile_weights=profile_weights,
        profile_values=profile_values,
        alpha_moment=alpha_moment,
        initial_guess=initial_guess,
        solver_tol=solver_tol,
        solver_maxfev=solver_maxfev,
    )


def theoretical_column_scaled_singular_value_curve(
    alpha: float,
    gamma: float,
    *,
    profile: ProfileSpec = "constant",
    s_max: float = 8.0,
    num_points: int = 161,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    profile_order: int = 128,
    solver_tol: float = 1e-10,
    solver_maxfev: int = 400,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> ColumnScaledTheoryCurve:
    """Prototype singular-value density for the column-scaled profile ansatz."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    if s_max <= 0.0:
        raise ValueError("s_max must be positive.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    if imag_eps <= 0.0:
        raise ValueError("imag_eps must be positive.")

    profile_nodes, profile_weights, profile_values, alpha_moment, profile_name = _profile_quadrature(
        alpha,
        profile,
        profile_order=profile_order,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    output_scale = _output_scale(alpha, entry_scale, normalization)
    singular_values = np.linspace(0.0, float(s_max), int(num_points))
    squared_singular_values = singular_values ** 2
    singular_density = np.zeros_like(singular_values)
    squared_density = np.full_like(singular_values, np.nan)
    y_row_values = np.full(singular_values.shape, np.nan + 1j * np.nan, dtype=complex)

    guess: Optional[complex] = None
    for idx in range(num_points - 1, 0, -1):
        s_out = singular_values[idx]
        s_base = s_out / output_scale
        z = complex(s_base, imag_eps)
        y_row = _solve_row_field_with_profile(
            alpha,
            gamma,
            z,
            quadrature_order=quadrature_order,
            profile_weights=profile_weights,
            profile_values=profile_values,
            alpha_moment=alpha_moment,
            initial_guess=guess,
            solver_tol=solver_tol,
            solver_maxfev=solver_maxfev,
        )
        h_val = _h_alpha_scalar(y_row, alpha, quadrature_order)
        sv_density_base = max(0.0, -2.0 * h_val.imag / (np.pi * s_base))
        singular_density[idx] = sv_density_base / output_scale
        squared_density[idx] = singular_density[idx] / (2.0 * s_out)
        y_row_values[idx] = y_row
        guess = y_row

    singular_density[0] = 0.0
    return ColumnScaledTheoryCurve(
        alpha=float(alpha),
        gamma=float(gamma),
        normalization=normalization,
        entry_scale=float(entry_scale),
        profile_name=profile_name,
        imag_eps=float(imag_eps),
        quadrature_order=int(quadrature_order),
        profile_order=int(profile_order),
        singular_values=singular_values,
        singular_density=singular_density,
        squared_singular_values=squared_singular_values,
        squared_density=squared_density,
        y_row=y_row_values,
        profile_nodes=profile_nodes,
        profile_values=profile_values,
        profile_alpha_moment=float(alpha_moment),
        atom_at_zero=float(1.0 - gamma),
    )


def sample_column_scaled_levy_matrix(
    n_rows: int,
    alpha: float,
    gamma: float,
    *,
    profile: ProfileSpec = "constant",
    entry_scale: float = 1.0,
    normalization: str = "stable",
    random_state: Optional[np.random.Generator] = None,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Sample a column-scaled rectangular heavy-tailed matrix."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    if n_rows < 2:
        raise ValueError("n_rows must be at least 2.")

    n_cols = max(1, int(round(gamma * n_rows)))
    u_cols, column_profile, profile_name = _profile_columns(
        n_cols,
        profile,
        constant_value=constant_value,
        left_value=left_value,
        right_value=right_value,
        split=split,
        exponent=exponent,
        cutoff=cutoff,
    )
    rng = random_state if random_state is not None else np.random.default_rng()
    matrix = stats.levy_stable.rvs(
        alpha,
        0.0,
        loc=0.0,
        scale=float(entry_scale),
        size=(n_rows, n_cols),
        random_state=rng,
    )
    matrix = np.asarray(matrix, dtype=float) * column_profile[None, :]
    divisor = (n_rows + n_cols) ** (1.0 / alpha)
    if normalization == "belinschi":
        divisor *= wl.belinschi_quantile_scale(alpha, entry_scale=entry_scale)
    return matrix / divisor, u_cols, column_profile, profile_name


def empirical_column_scaled_singular_value_spectrum(
    alpha: float,
    gamma: float,
    *,
    profile: ProfileSpec = "constant",
    n_rows: int = 96,
    num_matrices: int = 16,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    bins: int = 81,
    seed: Optional[int] = None,
    singular_range: Optional[tuple[float, float]] = None,
    squared_range: Optional[tuple[float, float]] = None,
    constant_value: float = 1.0,
    left_value: float = 0.75,
    right_value: float = 1.5,
    split: float = 0.5,
    exponent: float = 0.5,
    cutoff: float = 0.05,
) -> EmpiricalColumnScaledSpectrum:
    """Direct-SVD benchmark for the column-scaled singular-value law."""

    alpha = _validate_alpha(alpha)
    gamma = _validate_gamma(gamma)
    normalization = _validate_normalization(normalization)
    if num_matrices < 1:
        raise ValueError("num_matrices must be at least 1.")

    rng = np.random.default_rng(seed)
    singular_blocks = []
    squared_blocks = []
    profile_grid = None
    profile_values = None
    profile_name = None
    n_cols = max(1, int(round(gamma * n_rows)))
    gamma_eff = n_cols / n_rows

    for _ in range(num_matrices):
        matrix, profile_grid, profile_values, profile_name = sample_column_scaled_levy_matrix(
            n_rows,
            alpha,
            gamma,
            profile=profile,
            entry_scale=entry_scale,
            normalization=normalization,
            random_state=rng,
            constant_value=constant_value,
            left_value=left_value,
            right_value=right_value,
            split=split,
            exponent=exponent,
            cutoff=cutoff,
        )
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        singular_blocks.append(singular_values)
        squared_blocks.append(singular_values ** 2)

    all_singular = np.concatenate(singular_blocks)
    all_squared = np.concatenate(squared_blocks)
    sv_density, sv_edges = np.histogram(all_singular, bins=bins, range=singular_range, density=True)
    sq_density, sq_edges = np.histogram(all_squared, bins=bins, range=squared_range, density=True)
    return EmpiricalColumnScaledSpectrum(
        alpha=float(alpha),
        gamma=float(gamma_eff),
        n_rows=n_rows,
        n_cols=n_cols,
        normalization=normalization,
        entry_scale=float(entry_scale),
        profile_name=str(profile_name),
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
        column_profile_grid=np.asarray(profile_grid, dtype=float),
        column_profile_values=np.asarray(profile_values, dtype=float),
    )


def compare_constant_profile_to_wishart(
    alpha: float,
    gamma: float,
    *,
    s_max: float = 6.0,
    num_points: int = 121,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    profile_order: int = 128,
) -> dict[str, float]:
    """Compare the constant-profile prototype against ``wishart_levy.py``."""

    column_curve = theoretical_column_scaled_singular_value_curve(
        alpha,
        gamma,
        profile="constant",
        s_max=s_max,
        num_points=num_points,
        entry_scale=entry_scale,
        normalization=normalization,
        imag_eps=imag_eps,
        quadrature_order=quadrature_order,
        profile_order=profile_order,
        constant_value=1.0,
    )
    wishart_curve = wl.theoretical_singular_value_curve(
        alpha,
        gamma,
        s_max=s_max,
        num_points=num_points,
        entry_scale=entry_scale,
        normalization=normalization,
        imag_eps=imag_eps,
        quadrature_order=quadrature_order,
    )
    diff = np.nanmax(np.abs(column_curve.singular_density - wishart_curve.singular_density))
    rel = diff / max(float(np.nanmax(np.abs(wishart_curve.singular_density))), 1e-12)
    tail_diff = abs(
        singular_tail_prefactor(alpha, gamma, "constant", entry_scale=entry_scale, normalization=normalization)
        - wl.singular_tail_prefactor(alpha, gamma, entry_scale=entry_scale, normalization=normalization)
    )
    return {
        "max_abs_density_diff": float(diff),
        "max_rel_density_diff": float(rel),
        "tail_prefactor_diff": float(tail_diff),
    }
