"""Wigner-Lévy density of states following Cizeau-Bouchaud (1994).

This module implements the mean-field density-of-states workflow for the Lévy random
matrix ensemble introduced by Cizeau and Bouchaud (Phys. Rev. E 50, 1810,
1994). For the analytical mean-field solver we use the corrected, numerically stable
formulation of Burda et al. (Phys. Rev. E 75, 051126, 2007), which rewrites
the original real-axis cavity equations into a practical quadrature-based
fixed-point problem.

Phase 1 of the implementation covers:

* sampling symmetric Wigner-Lévy matrices with the paper's N^{-1 / mu} scaling,
* exact diagonalization and empirical spectral histograms,
* the corrected Cizeau-Bouchaud / Bouchaud-Cizeau mean-field DOS curve,
* save/load helpers for notebook-driven analysis,
* a light CLI for generating mean-field curves and empirical comparisons.

The localization / mobility-edge machinery from the later sections of the
paper is intentionally left for a later phase, but the exact-diagonalization
helpers already expose inverse participation ratios as a hook for that work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import integrate, special, stats
from scipy.optimize import root_scalar



@dataclass
class TheoryCurve:
    mu: float
    entry_scale: float
    z_prime: np.ndarray
    lambda_grid: np.ndarray
    density: np.ndarray
    c_eff: np.ndarray
    beta_eff: np.ndarray
    eta: np.ndarray
    burda_rescaling: float
    integration_method: str = "quad"
    mc_samples: Optional[int] = None
    mc_seed: Optional[int] = None


@dataclass
class EmpiricalSpectrum:
    mu: float
    matrix_size: int
    num_matrices: int
    entry_scale: float
    beta: float
    seed: Optional[int]
    eigenvalues: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    density: np.ndarray
    ipr: Optional[np.ndarray] = None


def _validate_mu(mu: float) -> float:
    mu = float(mu)
    if not (1.0 <= mu < 2.0):
        raise ValueError("mu must satisfy 1 <= mu < 2 for the implemented DOS solver.")
    return mu


def burda_rescaling(mu: float) -> float:
    """Rescaling from the C'=1 analytical normalization to entry-scale C=1.

    Burda et al. Eq. (45).
    """

    mu = _validate_mu(mu)
    numerator = special.gamma(1.0 + mu) * np.cos(np.pi * mu / 4.0)
    denominator = special.gamma(1.0 + mu / 2.0)
    return float((numerator / denominator) ** (1.0 / mu))


def mean_field_density_at_zero(mu: float, entry_scale: float = 1.0) -> float:
    """Closed-form DOS value at the origin.

    Burda et al. Eq. (47), with an extra 1 / entry_scale for matrix rescaling.
    """

    mu = _validate_mu(mu)
    prefactor = special.gamma(1.0 + 2.0 / mu) / np.pi
    bracket = (
        special.gamma(1.0 + mu / 2.0) ** 2 / special.gamma(1.0 + mu)
    ) ** (1.0 / mu)
    return float(prefactor * bracket / entry_scale)


def dos_tail_prefactor(mu: float, entry_scale: float = 1.0) -> float:
    """Prefactor of the large-|lambda| DOS tail.

    Using the Burda large-z' asymptotics together with the one-sided tail of the
    auxiliary stable law gives

        rho(lambda) ~ A_mu / |lambda|^{1 + mu},

    with A_mu = Gamma(1 + mu) * sin(pi mu / 2) / pi * entry_scale**mu.
    """

    mu = _validate_mu(mu)
    return float(
        special.gamma(1.0 + mu)
        * np.sin(np.pi * mu / 2.0)
        / np.pi
        * entry_scale ** mu
    )


def asymptotic_dos_tail(
    lambda_grid: np.ndarray,
    mu: float,
    *,
    entry_scale: float = 1.0,
) -> np.ndarray:
    """Large-|lambda| asymptotic DOS, A_mu / |lambda|^{1 + mu}."""

    amplitude = dos_tail_prefactor(mu, entry_scale=entry_scale)
    lam = np.asarray(lambda_grid, dtype=float)
    density = np.zeros_like(lam)
    mask = np.abs(lam) > 0.0
    density[mask] = amplitude / np.abs(lam[mask]) ** (1.0 + mu)
    return density



def inverse_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """IPR of eigenvectors, useful for the later localization phase."""

    return np.sum(np.abs(eigenvectors) ** 4, axis=0)


def sample_wigner_levy_matrix(
    size: int,
    mu: float,
    *,
    beta: float = 0.0,
    entry_scale: float = 1.0,
    random_state: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample a symmetric Wigner-Lévy matrix with N^{-1 / mu} scaling."""

    mu = _validate_mu(mu)
    if size < 2:
        raise ValueError("size must be at least 2.")

    rng = random_state if random_state is not None else np.random.default_rng()
    matrix = np.zeros((size, size), dtype=float)

    upper = np.triu_indices(size, k=1)
    off_diag = stats.levy_stable.rvs(
        mu,
        beta,
        loc=0.0,
        scale=entry_scale,
        size=upper[0].size,
        random_state=rng,
    )
    matrix[upper] = off_diag
    matrix[(upper[1], upper[0])] = off_diag

    diagonal = stats.levy_stable.rvs(
        mu,
        beta,
        loc=0.0,
        scale=entry_scale,
        size=size,
        random_state=rng,
    )
    matrix[np.diag_indices(size)] = diagonal
    matrix /= size ** (1.0 / mu)
    return matrix


def sample_eigensystem_ensemble(
    matrix_size: int,
    mu: float,
    *,
    num_matrices: int = 16,
    beta: float = 0.0,
    entry_scale: float = 1.0,
    compute_ipr: bool = False,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Sample eigenvalues from multiple exact-diagonalization runs."""

    mu = _validate_mu(mu)
    rng = np.random.default_rng(seed)
    eigenvalue_blocks = []
    ipr_blocks = [] if compute_ipr else None

    for _ in range(num_matrices):
        matrix = sample_wigner_levy_matrix(
            matrix_size, mu, beta=beta, entry_scale=entry_scale, random_state=rng,
        )
        if compute_ipr:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            ipr_blocks.append(inverse_participation_ratio(eigenvectors))
        else:
            eigenvalues = np.linalg.eigvalsh(matrix)
        eigenvalue_blocks.append(eigenvalues)

    all_eigenvalues = np.concatenate(eigenvalue_blocks)
    all_ipr = None if ipr_blocks is None else np.concatenate(ipr_blocks)
    return all_eigenvalues, all_ipr


def _monte_carlo_rule(samples: int, seed: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    if samples < 1:
        raise ValueError("mc_samples must be at least 1.")
    rng = np.random.default_rng(seed)
    uniforms = (np.arange(samples, dtype=float) + rng.random(samples)) / samples
    nodes = -np.log1p(-uniforms)
    weights = np.full(samples, 1.0 / samples, dtype=float)
    return nodes, weights



def _eta_integrals(
    mu: float,
    z_prime: float,
    eta: float,
    nodes: Optional[np.ndarray],
    weights: Optional[np.ndarray],
) -> tuple[float, float]:
    if nodes is None:
        cos_integral, _ = integrate.quad(
            lambda t: np.cos(t ** (2.0 / mu) * z_prime - eta * t) * np.exp(-t),
            0, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10,
        )
        sin_integral, _ = integrate.quad(
            lambda t: np.sin(t ** (2.0 / mu) * z_prime - eta * t) * np.exp(-t),
            0, np.inf, limit=200, epsabs=1e-12, epsrel=1e-10,
        )
        return float(cos_integral), float(sin_integral)
    phase = nodes ** (2.0 / mu) * z_prime - eta * nodes
    cos_integral = float(np.sum(weights * np.cos(phase)))
    sin_integral = float(np.sum(weights * np.sin(phase)))
    return cos_integral, sin_integral


def _solve_eta(
    mu: float,
    z_prime: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    *,
    eta_init: Optional[float] = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[float, float]:
    if z_prime == 0.0:
        cos_integral, _ = _eta_integrals(mu, 0.0, 0.0, nodes, weights)
        return 0.0, cos_integral

    eta_limit = np.tan(np.pi * mu / 4.0) * (1.0 - 1e-12)
    if eta_init is None:
        eta = np.sign(z_prime) * eta_limit * np.tanh(abs(z_prime) / (1.0 + abs(z_prime)))
    else:
        eta = float(np.clip(eta_init, -eta_limit, eta_limit))
    cos_integral = np.nan

    for _ in range(max_iter):
        cos_integral, sin_integral = _eta_integrals(mu, z_prime, eta, nodes, weights)
        if abs(cos_integral) < 1e-14:
            break
        update = np.clip(sin_integral / cos_integral, -eta_limit, eta_limit)
        new_eta = 0.5 * eta + 0.5 * update
        if abs(new_eta - eta) < tol:
            eta = new_eta
            cos_integral, _ = _eta_integrals(mu, z_prime, eta, nodes, weights)
            return float(eta), float(cos_integral)
        eta = new_eta

    def residual(value: float) -> float:
        cos_val, sin_val = _eta_integrals(mu, z_prime, value, nodes, weights)
        return sin_val / cos_val - value

    bracket = np.linspace(0.0, eta_limit, 65)
    values = np.array([residual(point) for point in bracket])
    sign_change = np.where(np.signbit(values[:-1]) != np.signbit(values[1:]))[0]
    if sign_change.size == 0:
        raise RuntimeError(f"Could not solve eta for z'={z_prime:.6g}.")

    candidates = []
    for idx in sign_change:
        left = bracket[idx]
        right = bracket[idx + 1]
        root = root_scalar(residual, bracket=(left, right), method="brentq")
        eta_candidate = float(root.root)
        cos_candidate, _ = _eta_integrals(mu, z_prime, eta_candidate, nodes, weights)
        candidates.append((eta_candidate, cos_candidate))

    positive_candidates = [item for item in candidates if item[1] > 0]
    pool = positive_candidates if positive_candidates else candidates
    eta, cos_integral = min(pool, key=lambda item: abs(item[0] - eta))
    return float(eta), float(cos_integral)


def asymptotic_tail_point(
    mu: float,
    z_prime: float,
    *,
    entry_scale: float,
) -> tuple[float, float, float, float, float]:
    eta_scale = np.tan(np.pi * mu / 4.0)
    eta_value = float(eta_scale * (1.0 - 1e-12))
    beta_value = float(1.0 - 1e-12)
    c_value = float(z_prime ** (-mu / 4.0))
    rescaling = burda_rescaling(mu) * entry_scale
    stable_scale = rescaling * (c_value ** (2.0 / mu))
    lambda_value = float(stable_scale * z_prime)
    density_value = float(
        dos_tail_prefactor(mu, entry_scale=entry_scale)
        / abs(lambda_value) ** (1.0 + mu)
    )
    return eta_value, c_value, beta_value, lambda_value, density_value


def _positive_theory_branch(
    mu: float,
    *,
    z_prime_max: float,
    num_points: int,
    entry_scale: float,
    integration_method: str,
    mc_samples: int,
    mc_seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if integration_method == "monte_carlo":
        nodes, weights = _monte_carlo_rule(mc_samples, mc_seed)
    else:
        nodes, weights = None, None
    z_prime = np.linspace(0.0, z_prime_max, num_points)

    eta = np.full_like(z_prime, np.nan)
    c_eff = np.full_like(z_prime, np.nan)
    beta_eff = np.full_like(z_prime, np.nan)
    lambda_grid = np.full_like(z_prime, np.nan)
    density = np.full_like(z_prime, np.nan)

    prefactor = (
        4.0
        / (np.pi * mu)
        * special.gamma(1.0 - mu / 2.0)
        * np.sin(np.pi * mu / 4.0)
    )
    eta_scale = np.tan(np.pi * mu / 4.0)
    rescaling = burda_rescaling(mu) * entry_scale
    alpha_self = mu / 2.0
    eta_prev = 0.0

    for idx, z_value in enumerate(z_prime):
        try:
            eta_value, cos_integral = _solve_eta(
                mu,
                float(z_value),
                nodes,
                weights,
                eta_init=eta_prev,
            )
            c_sq = prefactor * cos_integral
            if c_sq < -1e-12:
                continue
            c_value = float(np.sqrt(max(c_sq, 0.0)))
            beta_value = float(np.clip(eta_value / eta_scale, -1.0 + 1e-12, 1.0 - 1e-12))
            stable_scale = rescaling * (c_value ** (2.0 / mu))
            lambda_value = stable_scale * z_value
            density_value = float(
                stats.levy_stable.pdf(
                    lambda_value,
                    alpha_self,
                    beta_value,
                    loc=0.0,
                    scale=stable_scale,
                )
            )
        except RuntimeError:
            continue

        eta[idx] = eta_value
        c_eff[idx] = c_value
        beta_eff[idx] = beta_value
        lambda_grid[idx] = lambda_value
        density[idx] = density_value
        eta_prev = eta_value

    density[0] = mean_field_density_at_zero(mu, entry_scale=entry_scale)
    return z_prime, lambda_grid, density, c_eff, beta_eff, eta


def theoretical_wigner_levy_curve(
    mu: float,
    *,
    entry_scale: float = 1.0,
    z_prime_max: float = 12.0,
    num_points: int = 321,
    integration_method: str = "quad",
    mc_samples: int = 20000,
    mc_seed: Optional[int] = None,
) -> TheoryCurve:
    """Analytical mean-field DOS curve for the Wigner-Lévy ensemble.

    The curve is parameterized by the auxiliary variable z' used by Burda et al.
    The returned ``lambda_grid`` and ``density`` arrays are already expressed in
    the physical scaling of matrices whose entries are sampled with
    ``scipy.stats.levy_stable(..., scale=entry_scale)`` and then divided by
    ``N**(1 / mu)``.

    This evaluates the mean-field expression
    ``L_{mu/2}^{C(lambda), beta(lambda)}(lambda)`` rather than reconstructing
    the full density via the inverse Hilbert transform.
    """

    mu = _validate_mu(mu)
    if z_prime_max <= 0:
        raise ValueError("z_prime_max must be positive.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    if integration_method not in {"quad", "monte_carlo"}:
        raise ValueError("integration_method must be 'quad' or 'monte_carlo'.")

    z_pos, lam_pos, rho_pos, c_pos, beta_pos, eta_pos = _positive_theory_branch(
        mu,
        z_prime_max=z_prime_max,
        num_points=num_points,
        entry_scale=entry_scale,
        integration_method=integration_method,
        mc_samples=mc_samples,
        mc_seed=mc_seed,
    )

    z_full = np.concatenate((-z_pos[:0:-1], z_pos))
    lam_full = np.concatenate((-lam_pos[:0:-1], lam_pos))
    rho_full = np.concatenate((rho_pos[:0:-1], rho_pos))
    c_full = np.concatenate((c_pos[:0:-1], c_pos))
    beta_full = np.concatenate((-beta_pos[:0:-1], beta_pos))
    eta_full = np.concatenate((-eta_pos[:0:-1], eta_pos))

    # The z' parameterization can develop tiny finite-quadrature foldbacks, so
    # we sort by the physical lambda grid before exposing the curve.
    order = np.argsort(lam_full)
    z_full = z_full[order]
    lam_full = lam_full[order]
    rho_full = rho_full[order]
    c_full = c_full[order]
    beta_full = beta_full[order]
    eta_full = eta_full[order]

    return TheoryCurve(
        mu=mu,
        entry_scale=float(entry_scale),
        z_prime=z_full,
        lambda_grid=lam_full,
        density=rho_full,
        c_eff=c_full,
        beta_eff=beta_full,
        eta=eta_full,
        burda_rescaling=burda_rescaling(mu),
        integration_method=integration_method,
        mc_samples=mc_samples if integration_method == "monte_carlo" else None,
        mc_seed=mc_seed if integration_method == "monte_carlo" else None,
    )


def asymptotic_tail_curve(
    mu: float,
    *,
    entry_scale: float = 1.0,
    z_prime_max: float = 12.0,
    num_points: int = 321,
) -> TheoryCurve:
    """Mean-field DOS curve built entirely from the asymptotic tail formula.

    Returns a ``TheoryCurve`` with ``integration_method='asymptotic_tail'``.
    The z'=0 point is undefined and stored as NaN.  Use this as a separate
    overlay rather than as a fallback inside the self-consistent solver.
    """

    mu = _validate_mu(mu)
    z_pos = np.linspace(0.0, z_prime_max, num_points)

    eta_pos = np.full(num_points, np.nan)
    c_pos = np.full(num_points, np.nan)
    beta_pos = np.full(num_points, np.nan)
    lambda_pos = np.full(num_points, np.nan)
    density_pos = np.full(num_points, np.nan)

    for idx in range(1, num_points):
        eta_v, c_v, beta_v, lam_v, rho_v = asymptotic_tail_point(
            mu, float(z_pos[idx]), entry_scale=entry_scale,
        )
        eta_pos[idx] = eta_v
        c_pos[idx] = c_v
        beta_pos[idx] = beta_v
        lambda_pos[idx] = lam_v
        density_pos[idx] = rho_v

    z_full = np.concatenate((-z_pos[:0:-1], z_pos))
    lam_full = np.concatenate((-lambda_pos[:0:-1], lambda_pos))
    rho_full = np.concatenate((density_pos[:0:-1], density_pos))
    c_full = np.concatenate((c_pos[:0:-1], c_pos))
    beta_full = np.concatenate((-beta_pos[:0:-1], beta_pos))
    eta_full = np.concatenate((-eta_pos[:0:-1], eta_pos))

    order = np.argsort(lam_full)
    return TheoryCurve(
        mu=mu,
        entry_scale=float(entry_scale),
        z_prime=z_full[order],
        lambda_grid=lam_full[order],
        density=rho_full[order],
        c_eff=c_full[order],
        beta_eff=beta_full[order],
        eta=eta_full[order],
        burda_rescaling=burda_rescaling(mu),
        integration_method="asymptotic_tail",
        mc_samples=None,
        mc_seed=None,
    )


def empirical_wigner_levy_spectrum(
    mu: float,
    *,
    matrix_size: int = 96,
    num_matrices: int = 24,
    beta: float = 0.0,
    entry_scale: float = 1.0,
    bins: int = 101,
    compute_ipr: bool = False,
    seed: Optional[int] = None,
    histogram_range: Optional[tuple[float, float]] = None,
) -> EmpiricalSpectrum:
    """Exact-diagonalization benchmark for the DOS."""

    eigenvalues, ipr = sample_eigensystem_ensemble(
        matrix_size,
        mu,
        num_matrices=num_matrices,
        beta=beta,
        entry_scale=entry_scale,
        compute_ipr=compute_ipr,
        seed=seed,
    )
    density, bin_edges = np.histogram(eigenvalues, bins=bins, range=histogram_range, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return EmpiricalSpectrum(
        mu=float(mu),
        matrix_size=matrix_size,
        num_matrices=num_matrices,
        entry_scale=float(entry_scale),
        beta=float(beta),
        seed=seed,
        eigenvalues=eigenvalues,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        density=density,
        ipr=ipr,
    )
