"""One-sided heavy-tailed Wishart-Levy law (Theorem 2 of
``structured_wishart_levy.md``).

Dedicated solver for the case ``|tau(x, y)| = c(y)`` (column-only profile):
the structured field ``Y_r(x)`` collapses to a single scalar ``Y_r``, and the
density readout

    G_nu(zeta) = (1/zeta) h_alpha(Y_r(sqrt zeta)),
    rho_nu(t)  = -(1/(pi t)) Im h_alpha(Y_r(sqrt t)),

uses *independent-rule* ``h_alpha`` (its own Gauss-Laguerre integral, as in
``wishart_levy.py``). The structured solver uses the *shared-rule* ``h_alpha``
because the Theorem 1 collapse over the vector field ``Y_r(x)`` requires the
discrete identity ``h = 1 - (alpha/2) y g(y)`` to hold to machine precision
under quadrature; that constraint disappears at the scalar level. The
shared-rule numerical ``h`` has ~5e-3 absolute error in Im(h) that gets
amplified by the ``1/s`` factor in the density readout at small ``s``; the
independent-rule ``h`` is more accurate there.

Field equation (eq. 1 of ``one_sided_wishart_levy.md``):

    z^a Y_r = (gamma / (1 + gamma)) C_a int_0^1 c(v)^a g_a(
                  (C_a c(v)^a / ((1 + gamma) z^a)) g_a(Y_r)
              ) dv.

Tail (eq. 3 of the .md): ``f_SV(s) ~ B s^{-1-alpha}``,
``B = alpha gamma / (1 + gamma) * int_0^1 c^a * scale^a``.

Validation gates: ``compare_constant_to_wishart`` (Gate A,
``c == const`` -> wishart_levy to machine precision),
``compare_field_to_structured`` (Gate B, field ``Y_r`` matches
``structured_wishart_levy._scalar_one_sided_closure`` to machine
precision; density disagrees at small ``s`` in proportion to the
shared-vs-independent h gap -- the documented improvement).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy import optimize

import wishart_levy as wl
import structured_wishart_levy as swl


# c profile: a positive constant, a callable c(v) accepting a 1-D array, or
# a 1-D array sampled on a uniform grid in [0, 1].
OneSidedProfileSpec = Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]


@dataclass
class OneSidedTheoryCurve:
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
    y_row: np.ndarray            # shape (num_points,), complex scalar Y_r per z
    profile_alpha_moment: float  # int_0^1 c(v)^alpha dv
    atom_at_zero: float


# --- profile bookkeeping ---------------------------------------------------


def _profile_callable(c: OneSidedProfileSpec) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Normalise the profile spec to a callable returning |c(v)| on a 1-D array."""
    if isinstance(c, (int, float)):
        val = float(c)
        if not np.isfinite(val) or val < 0.0:
            raise ValueError("constant c must be finite and non-negative.")
        name = f"constant({val:.3g})"

        def cb(v: np.ndarray) -> np.ndarray:
            return np.full(np.shape(v), val, dtype=float)

        return cb, name
    if callable(c):

        def cb(v: np.ndarray) -> np.ndarray:
            out = np.asarray(c(np.asarray(v, dtype=float)), dtype=float)
            if out.shape != np.shape(v):
                raise ValueError("callable c must broadcast to its input shape.")
            return np.abs(out)

        return cb, "callable"
    if isinstance(c, np.ndarray):
        a = np.asarray(c, dtype=float).ravel()
        if a.size < 2:
            raise ValueError("array c must have at least 2 samples.")
        g = np.linspace(0.0, 1.0, a.size)

        def cb(v: np.ndarray) -> np.ndarray:
            return np.abs(np.interp(np.asarray(v, dtype=float), g, a))

        return cb, f"array[{a.size}]"
    raise ValueError("c must be a non-negative float, a callable, or a 1-D array.")


def profile_alpha_moment(
    c: OneSidedProfileSpec, alpha: float, *, profile_order: int = 96
) -> float:
    """int_0^1 c(v)^alpha dv via Gauss-Legendre."""
    alpha = wl._validate_alpha(alpha)
    cb, _ = _profile_callable(c)
    nodes, weights = swl._legendre01(int(profile_order))
    return float(np.sum(weights * cb(nodes) ** alpha))


# --- scalar field solver ---------------------------------------------------
#
# Same residual as ``swl._scalar_one_sided_closure`` (using ``swl._g_alpha_vec``
# for ``g_alpha``), but with seed priority anchor->wl_pair->asymptote (the
# structured solver does the opposite) and an imag-eps homotopy fallback for
# the stiff (small-s, alpha-near-2) corner where direct Newton finds a
# spurious branch.


def _scalar_residual(
    y_r: complex,
    *,
    z_alpha: complex,
    alpha: float,
    gamma: float,
    c_alpha_arr: np.ndarray,
    y_w: np.ndarray,
    quadrature_order: int,
) -> complex:
    C_a = wl.belinschi_constant(alpha)
    pref_outer = gamma * C_a / (1.0 + gamma)
    pref_inner = C_a / ((1.0 + gamma) * z_alpha)
    g_r = complex(swl._g_alpha_vec(np.array([y_r]), alpha, quadrature_order)[0])
    Y_c = pref_inner * c_alpha_arr * g_r
    g_c = swl._g_alpha_vec(Y_c, alpha, quadrature_order)
    integral = complex(np.sum(y_w * c_alpha_arr * g_c))
    return z_alpha * y_r - pref_outer * integral


def _solve_field_at_z(
    alpha: float,
    gamma: float,
    z: complex,
    *,
    c_alpha_arr: np.ndarray,
    y_w: np.ndarray,
    quadrature_order: int,
    anchor: Optional[complex] = None,
    tol: float = 1e-12,
    _homotopy: bool = True,
) -> complex:
    """Newton solve for the scalar Y_r at complex z; anchor-first."""
    z_alpha = z ** alpha
    kw = dict(z_alpha=z_alpha, alpha=alpha, gamma=gamma,
              c_alpha_arr=c_alpha_arr, y_w=y_w,
              quadrature_order=int(quadrature_order))

    def residual_real(v):
        y = complex(v[0], v[1])
        r = _scalar_residual(y, **kw)
        if not np.isfinite(r):
            return np.array([1e6, 1e6])
        return np.array([r.real, r.imag])

    seeds: list[complex] = []
    if anchor is not None and np.isfinite(anchor):
        seeds.append(complex(anchor))
    # Asymptote seed -- never empty.  Uses the clipped _g_alpha_vec engine
    # so no overflow warnings (wl.solve_y_pair was dropped from the backup
    # because its internal wl.g_alpha calls overflow at large-|y| trial
    # points; the anchor-first strategy plus asymptote covers continuation
    # without needing the Corollary-1 anchor at every z).
    g0 = complex(swl._g_alpha_vec(np.array([0.0]), alpha, int(quadrature_order))[0])
    seeds.append(complex(gamma * wl.belinschi_constant(alpha) * g0 /
                         ((1.0 + gamma) * z_alpha)))

    accept = max(tol, 1e-9)
    best = None
    best_r = np.inf
    for seed in seeds:
        sol = optimize.root(residual_real, [seed.real, seed.imag],
                            method="hybr", tol=tol)
        y = complex(sol.x[0], sol.x[1])
        r = abs(complex(_scalar_residual(y, **kw)))
        if sol.success and r < accept:
            return y
        if r < best_r:
            best_r, best = r, y

    # imag-eps homotopy: solve on a broadened (smoother) contour and sharpen
    # back, matching the structured solver's strategy at the stiff corner.
    if _homotopy and z.imag > 0:
        hseed = None
        chain_ok = True
        for fac in (64.0, 16.0, 4.0, 1.0):
            zz = complex(z.real, z.imag * fac)
            try:
                hseed = _solve_field_at_z(
                    alpha, gamma, zz, c_alpha_arr=c_alpha_arr, y_w=y_w,
                    quadrature_order=quadrature_order,
                    anchor=hseed, tol=tol, _homotopy=False,
                )
            except RuntimeError:
                chain_ok = False
                break
        if chain_ok and hseed is not None:
            r = abs(complex(_scalar_residual(hseed, **kw)))
            if r < accept:
                return hseed

    if best is not None and best_r < 1e-6:
        return best
    raise RuntimeError(
        f"one-sided field did not converge at z={z!r} "
        f"(best residual {best_r:.2e})"
    )


def solve_field(
    alpha: float,
    gamma: float,
    z: complex,
    *,
    c: OneSidedProfileSpec,
    profile_order: int = 64,
    quadrature_order: int = 96,
    anchor: Optional[complex] = None,
    tol: float = 1e-12,
) -> complex:
    """Solve the Theorem 2(i) scalar Y_r at one complex z (anchor-first)."""
    cb, _ = _profile_callable(c)
    y_nodes, y_w = swl._legendre01(int(profile_order))
    c_vals = cb(y_nodes)
    c_alpha_arr = np.abs(c_vals) ** alpha
    return _solve_field_at_z(
        alpha=alpha, gamma=gamma, z=z,
        c_alpha_arr=c_alpha_arr, y_w=y_w,
        quadrature_order=int(quadrature_order),
        anchor=anchor, tol=tol,
    )


# --- independent-rule h_alpha with overflow-safe clipping ------------------
#
# Same integrand as wishart_levy.h_alpha (its own Gauss-Laguerre rule, not
# the shared-with-g_alpha identity used by structured_wishart_levy), but
# with the np.clip guard from swl._g_alpha_vec so Newton/seed excursions
# into large-|y| territory don't overflow exp() and don't spam RuntimeWarnings.

_EXP_CLIP = 700.0


def _h_alpha_indep(y: complex, alpha: float, quadrature_order: int) -> complex:
    """Independent-rule h_alpha(y) = int_0^infty exp(-t) exp(-t^{a/2} y) dt,
    via Gauss-Laguerre with clipped exp argument."""
    nodes, weights = wl._laguerre_rule(int(quadrature_order))
    power = nodes ** (alpha / 2.0)
    expo = -power * complex(y)
    expo = np.clip(expo.real, -_EXP_CLIP, _EXP_CLIP) + 1j * expo.imag
    return complex(np.sum(weights * np.exp(expo)))


def _density_from_y(
    y: complex, s_base: float, alpha: float, quadrature_order: int,
) -> float:
    """rho(s) = -(2/(pi s)) Im h_alpha(Y_r(s)) -- Gram SV convention,
    using the clipped independent-rule h_alpha."""
    if s_base <= 0.0:
        return 0.0
    h = _h_alpha_indep(y, alpha, int(quadrature_order))
    val = -2.0 * h.imag / (np.pi * s_base)
    return max(0.0, val)


# --- main theory curve -----------------------------------------------------


def theoretical_one_sided_singular_value_curve(
    alpha: float,
    gamma: float,
    c: OneSidedProfileSpec,
    *,
    s_max: float = 8.0,
    num_points: int = 161,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    imag_eps: float = 1e-3,
    quadrature_order: int = 96,
    profile_order: int = 64,
    tol: float = 1e-12,
) -> OneSidedTheoryCurve:
    """Singular-value density of the one-sided heavy-tailed Wishart-Levy law."""
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    normalization = wl._validate_normalization(normalization)
    if s_max <= 0.0 or num_points < 3 or imag_eps <= 0.0:
        raise ValueError("need s_max>0, num_points>=3, imag_eps>0.")
    cb, name = _profile_callable(c)
    output_scale = wl._output_scale(alpha, entry_scale, normalization)

    # precompute the profile-quadrature arrays (Gauss-Legendre on [0, 1]) once
    y_nodes, y_w = swl._legendre01(int(profile_order))
    c_vals = cb(y_nodes)
    c_alpha_arr = np.abs(c_vals) ** alpha
    moment = float(np.sum(y_w * c_alpha_arr))

    sv = np.linspace(0.0, float(s_max), int(num_points))
    sv_density = np.zeros_like(sv)
    sq_density = np.full_like(sv, np.nan)
    y_row = np.full(int(num_points), np.nan + 1j * np.nan, dtype=complex)

    anchor: Optional[complex] = None
    n_failed = 0
    for idx in range(int(num_points) - 1, 0, -1):
        s_out = sv[idx]
        s_base = s_out / output_scale
        z = complex(s_base, imag_eps)
        try:
            y_r = _solve_field_at_z(
                alpha=alpha, gamma=gamma, z=z,
                c_alpha_arr=c_alpha_arr, y_w=y_w,
                quadrature_order=int(quadrature_order),
                anchor=anchor, tol=tol,
            )
        except RuntimeError:
            sv_density[idx] = np.nan
            sq_density[idx] = np.nan
            n_failed += 1
            continue
        anchor = y_r
        y_row[idx] = y_r
        d_base = _density_from_y(y_r, s_base, alpha, quadrature_order)
        sv_density[idx] = d_base / output_scale
        sq_density[idx] = sv_density[idx] / (2.0 * s_out) if s_out > 0.0 else np.nan

    # s = 0: continuous density is finite at the origin for gamma = 1 (Wigner /
    # plain-square corner, structured_wishart_levy.md Specialisations); for
    # gamma < 1 the (1 - gamma) atom lives there, not the continuous density.
    # We leave sv_density[0] = 0 as the "continuous part" convention.
    if n_failed > 0:
        import warnings
        warnings.warn(
            f"one-sided curve: {n_failed}/{int(num_points)} solver failures "
            f"(stiff corner; affected bins set to NaN).",
            RuntimeWarning, stacklevel=2,
        )
    return OneSidedTheoryCurve(
        alpha=float(alpha), gamma=float(gamma), normalization=normalization,
        entry_scale=float(entry_scale), profile_name=name,
        imag_eps=float(imag_eps), quadrature_order=int(quadrature_order),
        profile_order=int(profile_order),
        singular_values=sv, singular_density=sv_density,
        squared_singular_values=sv ** 2, squared_density=sq_density,
        y_row=y_row, profile_alpha_moment=float(moment),
        atom_at_zero=float(1.0 - gamma),
    )


# --- tail (Theorem 2(v) specialisation) ------------------------------------


def one_sided_singular_tail_prefactor(
    alpha: float,
    gamma: float,
    c: OneSidedProfileSpec,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    profile_order: int = 96,
    atomless: bool = False,
) -> float:
    """B in f_SV(s) ~ B s^{-1-alpha}, eq. (3) of one_sided_wishart_levy.md."""
    alpha = wl._validate_alpha(alpha)
    gamma = wl._validate_gamma(gamma)
    K = profile_alpha_moment(c, alpha, profile_order=profile_order)
    scale = wl._output_scale(alpha, entry_scale, normalization)
    g_factor = 1.0 if atomless else gamma
    return float(alpha * g_factor / (1.0 + gamma) * K * scale ** alpha)


# --- empirical sampler (Theorem 2 = column-scaled heavy-tailed matrix) -----


def sample_one_sided_levy_matrix(
    n_rows: int,
    alpha: float,
    gamma: float,
    c: OneSidedProfileSpec,
    *,
    entry_scale: float = 1.0,
    normalization: str = "stable",
    random_state: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one column-scaled heavy-tailed matrix M and return (M, c_grid, c_vals).

    Thin wrapper around ``swl.sample_structured_levy_matrix`` (with the
    deepest-common-ancestor rule: that function is already the right home for
    the SciPy <-> Belinschi sampling convention).  The 5-tuple structured
    return is collapsed to the natural 3-tuple of the one-sided case
    (M, column grid, column scales).
    """
    cb, _ = _profile_callable(c)
    M, _, y, T_profile, _ = swl.sample_structured_levy_matrix(
        n_rows, alpha, gamma, lambda _, Y: cb(Y),
        entry_scale=entry_scale, normalization=normalization,
        random_state=random_state,
    )
    # T_profile is shape (n_rows, n_cols); every row equals c_vals for tau(x,y)=c(y).
    return M, y, np.asarray(T_profile[0, :], dtype=float)


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
    """Gate A: c == 1 must reproduce ``wishart_levy``.

    Both sides use the independent-rule h_alpha here, so field and density
    agree to machine precision (no shared-vs-independent h offset).
    """
    wcur = wl.theoretical_singular_value_curve(
        alpha, gamma, s_max=s_max, num_points=num_points, imag_eps=imag_eps,
        quadrature_order=quadrature_order,
    )
    ocur = theoretical_one_sided_singular_value_curve(
        alpha, gamma, 1.0, s_max=s_max, num_points=num_points,
        imag_eps=imag_eps, quadrature_order=quadrature_order,
        profile_order=profile_order,
    )
    max_field = 0.0
    max_dens = 0.0
    for idx in range(num_points - 1, 0, -1):
        y_w = wcur.y1[idx]
        y_o = ocur.y_row[idx]
        if not (np.isfinite(y_w) and np.isfinite(y_o)):
            continue
        max_field = max(max_field, float(abs(y_w - y_o)))
        max_dens = max(max_dens,
                       float(abs(wcur.singular_density[idx]
                                 - ocur.singular_density[idx])))
    return {
        "max_field_diff": max_field,
        "max_density_diff": max_dens,
        "profile_alpha_moment": ocur.profile_alpha_moment,
    }


def compare_field_to_structured(
    alpha: float,
    gamma: float,
    c: OneSidedProfileSpec,
    *,
    s_max: float = 6.0,
    num_points: int = 41,
    imag_eps: float = 1e-2,
    quadrature_order: int = 64,
    profile_order: int = 48,
) -> dict[str, float]:
    """Gate B: field Y_r must match ``structured_wishart_levy`` to machine
    precision (same equation, same g_alpha rule); density values disagree at
    small s in proportion to shared- vs independent-rule h_alpha -- the
    documented improvement, reported here.
    """
    cb, _ = _profile_callable(c)
    scur = swl.theoretical_structured_singular_value_curve(
        alpha, gamma, lambda _, Y: cb(Y), s_max=s_max, num_points=num_points,
        imag_eps=imag_eps, quadrature_order=quadrature_order,
        profile_order=profile_order,
    )
    ocur = theoretical_one_sided_singular_value_curve(
        alpha, gamma, c, s_max=s_max, num_points=num_points,
        imag_eps=imag_eps, quadrature_order=quadrature_order,
        profile_order=profile_order,
    )
    max_field = 0.0
    max_dens_small = 0.0
    max_dens_bulk = 0.0
    for idx in range(num_points - 1, 0, -1):
        Yr_s = scur.y_row[idx]
        y_o = ocur.y_row[idx]
        if not (np.all(np.isfinite(Yr_s)) and np.isfinite(y_o)):
            continue
        max_field = max(max_field, float(np.max(np.abs(Yr_s - y_o))))
        s_out = ocur.singular_values[idx]
        d_o = ocur.singular_density[idx]
        d_s = scur.singular_density[idx]
        if s_out < 0.5:
            max_dens_small = max(max_dens_small, float(abs(d_o - d_s)))
        else:
            max_dens_bulk = max(max_dens_bulk, float(abs(d_o - d_s)))
    return {
        "max_field_diff": max_field,
        "max_density_diff_small_s": max_dens_small,
        "max_density_diff_bulk_s": max_dens_bulk,
    }
