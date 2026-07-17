"""Cavity growth-rate localisation criterion (companion to localisation.md).

Localisation of the singular vectors of a structured heavy-tailed matrix is
the stability of the Im G = 0 fixed point of the bipartite cavity recursion
(localisation.md sec. 2).  This module measures that stability directly by
population dynamics: propagate a tangent (linearised) imaginary channel dIm
through the real-part cavity pool and read off

    phi(s) = per-sweep growth rate of the mean log imaginary part,

the quenched Lyapunov exponent of the linearised recursion.  phi(s) > 0 is
delocalised, phi(s) < 0 localised; the mobility edge s^* is the zero of
phi.  No transfer-operator kernel, no moment family: the criterion is the
growth rate itself.

Finite pools bias phi downward (front-velocity selection), with the
Brunet-Derrida form phi_P = phi_inf - c / ln^2 P.  Each s-point therefore
runs a doubling ladder over pool sizes RUNGS (duplicating the pool as a warm
start, short re-burn per rung) and quotes both the per-rung phi (any fixed-P
value is a lower bound) and the BD-extrapolated phi_inf.

Amplitude convention (CMS physical units): neighbour amplitudes are
((2K)^{-1/alpha} x)^2 with x exact symmetric alpha-stable draws, so s is in
physical singular-value units at every alpha.  The Jacobian cell uses the
quenched row profile chi_i = sigma_W |phi'((q*)^{1/alpha} Z_i)| at the
forward fixed point q* (ht_mlp_jacobian.md sec. 6).

Validation is against direct diagonalisation of the Hermitised Jacobian
(empirical singular vectors), not against any closed-form oracle.

Anchor gate (2026-07-17), cell (alpha, sigma_W) = (1.5, 1.0): the direct
mean-log readout gives s* = 4.67 at the top rung (P = 2e4) and 6.09
BD-extrapolated, matching the retired frozen-moment readout on the same
recursion (4.67-4.71 / ~5.9 +- 0.5) -- the two coincide in the pool limit.
Per-cell cost ~40 min on one core at production settings.

Run:
  python localisation.py probe                    # anchor cell (1.5, 1.0)
  python localisation.py cell alpha 1.5 sigma_W 1.0
  PHI_SMOKE=1 python localisation.py probe        # fast smoke settings
Timer-instrumented per CLAUDE.md.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
from contextlib import contextmanager
from math import pi
from pathlib import Path
from time import time

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RUNGS = (2_500, 5_000, 10_000, 20_000)
K_NB = 256
BURN0, REBURN = 60, 20
SWEEPS_R, SWEEPS_TOP = 120, 150
N_S = 14
BUF = 4_000_000

if os.environ.get("PHI_SMOKE"):
    RUNGS = (500, 1_000)
    BURN0, REBURN, SWEEPS_R, SWEEPS_TOP = 15, 8, 25, 25
    N_S = 4
    BUF = 400_000


@contextmanager
def Timer(label, log=None):
    tic = time()
    yield
    dt = time() - tic
    print(f"[{label}] elapsed {dt:.3f}s", flush=True)
    if log is not None:
        log.append((label, dt))


def cms_stable(alpha, size, rng):
    """Symmetric alpha-stable, CF exp(-|t|^alpha) (Chambers-Mallows-Stuck)."""
    U = rng.uniform(-pi / 2, pi / 2, size)
    W = rng.exponential(1.0, size)
    return (np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)
            * (np.cos(U - alpha * U) / W) ** ((1.0 - alpha) / alpha))


def _sweeps(Gr, Gc, dGr, dGc, s, chi2, wbuf, rng, burn, sweeps):
    """Run burn + sweeps pool refreshes at the current rung size.

    Each sweep updates the real-part cavity pool (row then column leg) and
    propagates the tangent dIm through the same draws, then renormalises the
    tangent by its geometric mean g.  log g is the per-sweep growth of the
    mean log imaginary part; phi at this rung is its average over the
    post-burn sweeps.  Returns (phi, updated state).
    """
    P = Gr.shape[0]
    c2 = chi2[:P]
    logs = []
    for t in range(burn + sweeps):
        amp = wbuf[rng.integers(0, BUF, (P, K_NB))]
        nb = rng.integers(0, P, (P, K_NB))
        Gr = 1.0 / (s - c2 * (amp * Gc[nb]).sum(axis=1))
        dGr = Gr ** 2 * c2 * (amp * dGc[nb]).sum(axis=1)
        amp = wbuf[rng.integers(0, BUF, (P, K_NB))]
        nb = rng.integers(0, P, (P, K_NB))
        Gc = 1.0 / (s - (amp * c2[nb] * Gr[nb]).sum(axis=1))
        dGc = Gc ** 2 * (amp * c2[nb] * dGr[nb]).sum(axis=1)
        g = np.exp(np.mean(np.log(np.clip(dGr, 1e-300, None))))
        dGr /= g
        dGc /= g
        if t >= burn:
            logs.append(np.log(g))
    return float(np.mean(logs)), Gr, Gc, dGr, dGc


def growth_rate_ladder(s, alpha, chi2, wbuf, rng):
    """phi(s) at every pool rung (doubling schedule, warm starts).

    chi2 has size RUNGS[-1]; rung P uses its first P entries (quenched
    profile), so doubled (G, dIm) pairs land on fresh chi values and the
    short re-burn re-equilibrates them.  Returns array of phi over RUNGS.
    """
    P0 = RUNGS[0]
    Gr = np.full(P0, -1.0 / s)
    Gc = np.full(P0, -1.0 / s)
    dGr = np.ones(P0)
    dGc = np.ones(P0)
    out = np.empty(len(RUNGS))
    for r, P in enumerate(RUNGS):
        if r > 0:
            Gr = np.concatenate([Gr, Gr])
            Gc = np.concatenate([Gc, Gc])
            dGr = np.concatenate([dGr, dGr])
            dGc = np.concatenate([dGc, dGc])
        burn = BURN0 if r == 0 else REBURN
        sweeps = SWEEPS_TOP if r == len(RUNGS) - 1 else SWEEPS_R
        out[r], Gr, Gc, dGr, dGc = _sweeps(
            Gr, Gc, dGr, dGc, s, chi2, wbuf, rng, burn, sweeps)
    return out


def bd_extrapolate(phi_rungs):
    """phi_inf from phi_P = phi_inf - c/ln^2 P (least squares over rungs)."""
    x = 1.0 / np.log(np.array(RUNGS, dtype=float)) ** 2
    slope, intercept = np.polyfit(x, phi_rungs, 1)
    return float(intercept), float(-slope)


def zero_crossing(s_grid, phi):
    """First descending zero of phi(s), log-s interpolated."""
    for i in range(len(s_grid) - 1):
        if phi[i] > 0 >= phi[i + 1]:
            f = phi[i] / (phi[i] - phi[i + 1] + 1e-300)
            return float(np.exp(np.log(s_grid[i])
                                + f * (np.log(s_grid[i + 1])
                                       - np.log(s_grid[i]))))
    return np.nan


def cell(args):
    """One (alpha, sigma_W) Jacobian cell: phi(s) ladder + BD zero s^*.

    Deterministic per-cell rng, so results are host-independent.  Returns
    (alpha, sigma_w, s_star_inf, s_star_rungs, s_grid, phi_inf, phi_rungs,
    q_star, elapsed).
    """
    alpha, sigma_w = args
    t0 = time()
    import torch
    import RMT
    torch.manual_seed(0)
    q = float(RMT.q_star_MC(alpha, sigma_w)[-1])
    rng = np.random.default_rng(
        hash((round(alpha * 100), round(sigma_w * 100))) % 2**32)
    z = 2 ** (-1 / alpha) * cms_stable(alpha, RUNGS[-1], rng)
    chi2 = (sigma_w / np.cosh(q ** (1 / alpha) * z) ** 2) ** 2
    scale = (2.0 * K_NB) ** (-1.0 / alpha)
    wbuf = (scale * cms_stable(alpha, BUF, rng)) ** 2
    s_grid = np.geomspace(0.8 * sigma_w, 12.0 * sigma_w, N_S)
    phi_rungs = np.empty((len(RUNGS), N_S))
    phi_inf = np.empty(N_S)
    for i, s in enumerate(s_grid):
        phi_rungs[:, i] = growth_rate_ladder(float(s), alpha, chi2, wbuf,
                                             rng)
        phi_inf[i], _ = bd_extrapolate(phi_rungs[:, i])
    s_star_inf = zero_crossing(s_grid, phi_inf)
    s_star_rungs = np.array([zero_crossing(s_grid, phi_rungs[r])
                             for r in range(len(RUNGS))])
    return (alpha, sigma_w, s_star_inf, s_star_rungs, s_grid, phi_inf,
            phi_rungs, float(q), time() - t0)


def report_cell(a, sw):
    t0 = time()
    (alpha, sigma_w, s_inf, s_rungs, s_grid, phi_inf, phi_rungs, q,
     dt) = cell((a, sw))
    print(f"cell ({a}, {sw}): q*={q:.4f}  elapsed {dt:.0f}s")
    hdr = "  ".join(f"P={P}" for P in RUNGS)
    print(f"   s     | phi at {hdr} | extrap")
    for i, s in enumerate(s_grid):
        row = "  ".join(f"{phi_rungs[r, i]:+.4f}" for r in range(len(RUNGS)))
        print(f"  s={s:6.2f} | {row} | {phi_inf[i]:+.4f}")
    rungs_s = "/".join("None" if np.isnan(v) else f"{v:.3f}" for v in s_rungs)
    print(f"  s* per rung: {rungs_s}")
    print(f"  s*_inf = {s_inf:.3f}")
    print(f"[cell total] elapsed {time()-t0:.0f}s")
    print("CELL-DONE")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if mode == "probe":
        report_cell(1.5, 1.0)
    elif mode == "cell":
        kw = dict(zip(sys.argv[2::2], sys.argv[3::2]))
        report_cell(float(kw["alpha"]), float(kw["sigma_W"]))
    else:
        raise SystemExit(__doc__)
