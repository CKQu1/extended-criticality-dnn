"""Mobility edge of symmetric Levy matrices (Tarquini-Biroli-Tarzia).

Numerical solver for the closed mobility-edge equation of Wigner-Levy matrices
(arXiv 1507.00296 / PRL 116, 010601). Derivation: RMT/levy_mobility_edge.md.
Literature note: .agents/notes/tarquini-2015.md.

Three steps:
  1. solve_C_beta   -- self-consistent (C(E), beta(E)) by population dynamics.
  2. ell_plus       -- the oscillatory integral ell(E).
  3. mobility_determinant / find_mobility_edge -- D(E) = 0 (m = 1/2).

CLI:
  python levy_mobility_edge.py edge mu 0.5            # E* at one mu
  python levy_mobility_edge.py validate               # full validation + figure
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from math import cos, gamma as gammafn, pi, sin, tan
from time import time
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Timer (per CLAUDE.md: time subparts so bottlenecks are visible at a glance).
# ---------------------------------------------------------------------------


@contextmanager
def Timer(label: str, log: Optional[list] = None):
    tic = time()
    yield
    dt = time() - tic
    print(f"[{label}] elapsed {dt:.3f}s")
    if log is not None:
        log.append((label, dt))


# ---------------------------------------------------------------------------
# Chambers-Mallows-Stuck sampler for a standard stable (S1 parameterisation).
#
# Returns X with characteristic function (alpha != 1, unit scale, zero shift)
#     E[e^{i k X}] = exp(-|k|^alpha (1 - i b sign(k) tan(pi alpha/2))).
# ---------------------------------------------------------------------------


def cms_stable(alpha: float, b: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < alpha < 1.0):
        raise ValueError("cms_stable specialised to alpha in (0, 1) (= mu/2).")
    U = rng.uniform(-pi / 2.0, pi / 2.0, size=size)
    W = rng.exponential(1.0, size=size)
    zeta = b * tan(pi * alpha / 2.0)
    B = np.arctan(zeta) / alpha
    Sscale = (1.0 + zeta * zeta) ** (1.0 / (2.0 * alpha))
    X = (
        Sscale
        * np.sin(alpha * (U + B))
        / np.cos(U) ** (1.0 / alpha)
        * (np.cos(U - alpha * (U + B)) / W) ** ((1.0 - alpha) / alpha)
    )
    return X


def sample_S(C: float, beta: float, mu: float, size: int,
             rng: np.random.Generator) -> np.ndarray:
    """Sample the physical self-energy real part S, scale C, skewness beta.

    S = sum_j h_j^2 Re G_j is dominated by positive Re G in the bulk, hence its
    heavy tail is positive-skewed: in the standard S1 parameterisation that is
    b = +beta (right tail heavier). The CF of this physical S is therefore

        E[e^{ikS}] = exp(-C k^{mu/2} (1 - i beta tan(pi mu/4))),   k > 0,

    i.e. the *minus* sign (see ell_plus). NOTE: TBT write their characteristic
    function L-hat with a +i beta sign; sampling with b = -beta (to match their
    written CF literally) collapses to the spurious one-sided fixed point
    beta = 1. The b = +beta sampler is validated convention-free against the
    LePage cavity sum (both give identical C, beta). Verified by selftest_cms.
    """
    a = mu / 2.0
    gamma = C ** (1.0 / a)
    return gamma * cms_stable(a, beta, size, rng)


# ---------------------------------------------------------------------------
# Step 1: self-consistent (C(E), beta(E)).
# ---------------------------------------------------------------------------


def solve_C_beta(
    E: float,
    mu: float,
    pool: int = 400_000,
    iters: int = 80,
    burn: int = 60,
    damping: float = 0.5,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[float, float]:
    a = mu / 2.0
    pref = gammafn(1.0 - a) * cos(pi * mu / 4.0)
    rng = np.random.default_rng(seed)
    C, beta = 1.0, 0.0
    Cs, betas = [], []
    for it in range(iters):
        S = sample_S(C, beta, mu, pool, rng)
        ReG = 1.0 / (E - S)
        absp = np.abs(ReG) ** a
        m0 = absp.mean()
        C_new = pref * m0
        beta_new = float((absp * np.sign(ReG)).mean() / m0)
        beta_new = max(-1.0, min(1.0, beta_new))
        C = (1.0 - damping) * C + damping * C_new
        beta = (1.0 - damping) * beta + damping * beta_new
        if it >= burn:
            Cs.append(C)
            betas.append(beta)
        if verbose:
            print(f"  it={it:3d}  C={C:.6f}  beta={beta:+.6f}")
    return float(np.mean(Cs)), float(np.mean(betas))


# ---------------------------------------------------------------------------
# Shared oscillatory Fourier integral of the closed-form stable CF.
#
#   I(kpow, phase_sign) = int_0^inf k^{kpow} exp(-C k^a)
#                           exp(i (kE + phase_sign * C beta t k^a)) dk,
#   a = mu/2, t = tan(pi mu/4).
#
# The fast oscillation e^{ikE} decays only via exp(-C k^a) (slow for small mu)
# and reaches huge k, so the tail k in [1, inf) uses the oscillatory QAWF rule
# (scipy weight='cos'/'sin'); the head k in [0, 1] carries the integrable
# k^{kpow} singularity (kpow > -1) and uses ordinary adaptive quad.
# ---------------------------------------------------------------------------


def _fourier_integral(E: float, mu: float, C: float, beta: float,
                      kpow: float, phase_sign: float) -> complex:
    from scipy.integrate import quad

    a = mu / 2.0
    t = tan(pi * mu / 4.0)

    def env(k):
        return k ** kpow * np.exp(-C * k ** a)

    def ph(k):
        return phase_sign * C * beta * t * k ** a

    R_head = quad(lambda k: env(k) * np.cos(k * E + ph(k)), 0.0, 1.0, limit=400)[0]
    I_head = quad(lambda k: env(k) * np.sin(k * E + ph(k)), 0.0, 1.0, limit=400)[0]

    gr = lambda k: env(k) * np.cos(ph(k))  # noqa: E731
    gi = lambda k: env(k) * np.sin(ph(k))  # noqa: E731
    R_tail = (quad(gr, 1.0, np.inf, weight="cos", wvar=E, limit=300)[0]
              - quad(gi, 1.0, np.inf, weight="sin", wvar=E, limit=300)[0])
    I_tail = (quad(gr, 1.0, np.inf, weight="sin", wvar=E, limit=300)[0]
              + quad(gi, 1.0, np.inf, weight="cos", wvar=E, limit=300)[0])

    return complex((R_head + R_tail) + 1j * (I_head + I_tail))


# ---------------------------------------------------------------------------
# Step 1 (deterministic): self-consistent (C, beta) by k-space fixed point.
#
# The TBT self-consistency integrals (Cizeau-Bouchaud form) are convolutions of
# the stable density with |E-S|^{-mu/2}; the Fourier transform of |x|^{-a}
# (prop. |k|^{a-1}) turns them into a single complex k-integral
#     J_c(E) = int_0^inf k^{a-1} e^{ikE} conj(Lhat(k)) dk,
# with conj(Lhat) = exp(-C k^a (1 + i beta t)) (phase_sign = -1), and
#     C    = Gamma(1-a)^2 sin(pi a) / pi * Re J_c,
#     beta = cot(pi a/2) * Im J_c / Re J_c.
# Deterministic (no Monte-Carlo noise); ~40 iterations. Cross-checked against
# the population-dynamics solver solve_C_beta and the LePage cavity sum.
# ---------------------------------------------------------------------------


def solve_C_beta_det(E: float, mu: float, iters: int = 400,
                     damping: float = 0.4, tol: float = 1e-11) -> Tuple[float, float]:
    a = mu / 2.0
    pre_C = gammafn(1.0 - a) ** 2 * sin(pi * a) / pi
    cot = cos(pi * a / 2.0) / sin(pi * a / 2.0)
    C, beta = 1.0, 0.0
    for _ in range(iters):
        Jc = _fourier_integral(E, mu, C, beta, a - 1.0, -1.0)
        C_new = max(pre_C * Jc.real, 1e-8)
        beta_new = max(-1.0, min(1.0, cot * Jc.imag / Jc.real))
        nC = (1.0 - damping) * C + damping * C_new
        nb = (1.0 - damping) * beta + damping * beta_new
        if abs(nC - C) < tol and abs(nb - beta) < tol:
            C, beta = nC, nb
            break
        C, beta = nC, nb
    return C, beta


# ---------------------------------------------------------------------------
# Step 2: ell(E) = (1/pi) int_0^inf k^{mu-1} Lhat(k) e^{ikE} dk.
# Physical CF of S (S1 b=+beta, LePage-confirmed): Lhat = exp(-C k^a (1 - i beta t)),
# i.e. phase_sign = +1.
# ---------------------------------------------------------------------------


def ell_plus(E: float, mu: float, C: float, beta: float) -> complex:
    return _fourier_integral(E, mu, C, beta, mu - 1.0, +1.0) / pi


def ell_converged(E: float, mu: float, C: float, beta: float) -> complex:
    return ell_plus(E, mu, C, beta)


# ---------------------------------------------------------------------------
# Step 3: mobility-edge determinant and root.
# ---------------------------------------------------------------------------


def _coeffs(mu: float) -> Tuple[float, float]:
    if mu >= 1.0:
        # K_mu = mu Gamma(1/2 - mu/2)^2/2 diverges only at mu=1; it is finite on
        # (1, 2). But TBT's equation has no physical solution for mu >= 1 (BG
        # delocalisation; cf. localisation.md sec. 4), so return inf to make
        # find_mobility_edge yield no edge.
        return float("inf"), sin(pi * mu / 2.0)
    K = mu * gammafn(0.5 - mu / 2.0) ** 2 / 2.0
    s = sin(pi * mu / 2.0)
    return K, s


def mobility_determinant(E: float, mu: float, C: float, beta: float) -> float:
    K, s = _coeffs(mu)
    ell = ell_converged(E, mu, C, beta)
    return float(K * K * (s * s - 1.0) * abs(ell) ** 2 - 2.0 * s * K * ell.real + 1.0)


def D_of_E(E: float, mu: float, **cb_kw) -> float:
    C, beta = solve_C_beta_det(E, mu, **cb_kw)
    return mobility_determinant(E, mu, C, beta)


def find_mobility_edge(
    mu: float,
    E_grid: Optional[np.ndarray] = None,
    refine_tol: float = 1e-2,
    cb_kw: Optional[dict] = None,
    verbose: bool = True,
) -> Optional[float]:
    """Scan D(E) over E_grid, bracket the sign change, bisect to E*."""
    cb_kw = cb_kw or {}
    if E_grid is None:
        E_grid = np.arange(0.5, 12.01, 0.5)
    Ds = []
    for E in E_grid:
        D = D_of_E(float(E), mu, **cb_kw)
        Ds.append(D)
        if verbose:
            print(f"  mu={mu:.2f}  E={E:6.3f}  D={D:+.5f}")
    Ds = np.array(Ds)
    sign = np.sign(Ds)
    idx = np.where(sign[:-1] < sign[1:])[0]  # - to + crossing
    if len(idx) == 0:
        return None
    i = idx[-1]
    lo, hi = float(E_grid[i]), float(E_grid[i + 1])
    Dlo = Ds[i]
    while hi - lo > refine_tol:
        mid = 0.5 * (lo + hi)
        Dm = D_of_E(mid, mu, **cb_kw)
        if (Dlo < 0) == (Dm < 0):
            lo, Dlo = mid, Dm
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Self-test: CMS sampler reproduces the TBT characteristic function.
# ---------------------------------------------------------------------------


def selftest_cms(mu: float = 0.5, C: float = 1.3, beta: float = 0.4,
                 size: int = 2_000_000, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    S = sample_S(C, beta, mu, size, rng)
    a = mu / 2.0
    for k in (0.3, 0.7, 1.5):
        emp = np.mean(np.exp(1j * k * S))
        # physical CF: minus sign on the i beta term (S1 b=+beta).
        target = np.exp(-C * k ** a * (1.0 - 1j * beta * tan(pi * mu / 4.0)))
        err = abs(emp - target)
        print(f"  CF k={k:.2f}: emp={emp.real:+.4f}{emp.imag:+.4f}j  "
              f"target={target.real:+.4f}{target.imag:+.4f}j  |err|={err:.4f}")
        assert err < 0.02, f"CMS CF mismatch at k={k}: |err|={err:.4f}"
    print("  selftest_cms OK")


# ---------------------------------------------------------------------------
# Validation driver.
# ---------------------------------------------------------------------------


def run_validation() -> None:
    log: list = []
    with Timer("selftest_cms", log):
        selftest_cms()

    cb: dict = {}  # deterministic solver defaults

    with Timer("E*(0.5)", log):
        Estar_half = find_mobility_edge(0.5, E_grid=np.arange(2.5, 5.01, 0.5),
                                        cb_kw=cb)
    print(f"\nmu=0.5: E* = {Estar_half}  (TBT target ~ 3.85)\n")

    mus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    edges = {}
    with Timer("E*(mu) sweep", log):
        for mu in mus:
            hi = 7.0 if mu < 0.65 else (14.0 if mu < 0.85 else 26.0)
            grid = np.arange(1.0, hi + 0.01, 1.0)
            Es = find_mobility_edge(mu, E_grid=grid, cb_kw=cb, verbose=False)
            edges[mu] = Es
            print(f"  mu={mu:.2f}  E* = {Es}")

    with Timer("no-edge check mu>=1", log):
        for mu in (1.0, 1.2):
            grid = np.arange(1.0, 20.01, 2.0)
            Es = find_mobility_edge(mu, E_grid=grid, cb_kw=cb, verbose=False)
            print(f"  mu={mu:.2f}  E* = {Es}  (expected None)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os

        ms = [m for m in mus if edges[m] is not None]
        es = [edges[m] for m in ms]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(ms, es, "o-", label="solver E*(mu)")
        ax.plot([0.5], [3.85], "r*", ms=14, label="TBT mu=0.5: 3.85")
        ax.set_xlabel("mu"); ax.set_ylabel("E*")
        ax.set_title("Levy-matrix mobility edge (TBT eq:mobility)")
        ax.legend(); ax.grid(alpha=0.3)
        outdir = "fig/levy_mobility_edge"
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "phase_diagram.png")
        fig.tight_layout(); fig.savefig(path, dpi=130)
        print(f"\nsaved {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")

    print("\ntimings:")
    for label, dt in log:
        print(f"  {label:24s} {dt:8.3f}s")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse(argv):
    kw = {}
    i = 0
    while i + 1 < len(argv):
        kw[argv[i]] = float(argv[i + 1])
        i += 2
    return kw


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "validate":
        run_validation()
    elif cmd == "selftest":
        selftest_cms()
    elif cmd == "edge":
        kw = _parse(sys.argv[2:])
        mu = kw.pop("mu", 0.5)
        with Timer(f"E*({mu})"):
            Es = find_mobility_edge(mu)
        print(f"mu={mu}: E* = {Es}")
    else:
        print(__doc__)
