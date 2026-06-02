"""Solvers and validations for RMT/localisation.md.

Part 1 (md sec. 3.6) -- structured one-sided Levy mobility edge, the closed-form
eq. (7): two-sided real-part closure + profile-integrated L_R + the 2x2
Perron condition. Deterministic. Reduces to levy_mobility_edge.py at a constant
profile (reduction gate). Valid only for mu = alpha < 1 (md sec. 3.7); for
alpha > 1 it returns continuation artifacts -- use Part 2 there.

Part 2 (md sec. 3.7) -- the general Tarquini imaginary-part-stability criterion
by complex cavity population dynamics. Valid for any alpha. The eta-scaling
exponent  p = d log(Im G_typ) / d log eta  is 0 (delocalised) or 1 (localised);
the mobility edge is where p crosses 1/2. This is the robust evaluation of "the
sec. 3.7 integral operator's Perron eigenvalue = 1", and it delivers the
profile-sparsification edge for the heavy-tailed MLP Jacobian at alpha in (1,2).

Run:
  python localisation.py gate       # Part 1 reduction gate (alpha=0.5 -> 3.29)
  python localisation.py cavity      # Part 2 unstructured vs saturating profile
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from math import cos, gamma as gammafn, log, pi, sin
from time import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import levy_mobility_edge as lme  # noqa: E402


@contextmanager
def Timer(label, log=None):
    tic = time()
    yield
    dt = time() - tic
    print(f"[{label}] elapsed {dt:.3f}s", flush=True)
    if log is not None:
        log.append((label, dt))


def _gl01(n):
    """Gauss-Legendre nodes/weights on [0, 1]."""
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def symmetric_stable(alpha, size, rng):
    """Standardised symmetric alpha-stable sample (Chambers-Mallows-Stuck)."""
    U = rng.uniform(-pi / 2.0, pi / 2.0, size)
    W = rng.exponential(1.0, size)
    return (np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha)
            * (np.cos((1.0 - alpha) * U) / W) ** ((1.0 - alpha) / alpha))


# ---------------------------------------------------------------------------
# Part 1: structured one-sided mobility edge (eq. 7), valid mu = alpha < 1.
# ---------------------------------------------------------------------------


def solve_two_sided(E, alpha, c_vals, c_w, iters=200, damping=0.4, tol=1e-10):
    """Coupled real-part closure (C_R, beta_R, C_C, beta_C); md eq. (3), sec. 3.6.

    Row self-energy at profile a is alpha/2-stable, scale a^alpha C_R; column
    self-energy is profile-averaged, scale C_C. Reuses levy_mobility_edge's
    k-space moment integral _fourier_integral.
    """
    a = alpha / 2.0
    pre_C = gammafn(1.0 - a) ** 2 * sin(pi * a) / pi
    cot = cos(pi * a / 2.0) / sin(pi * a / 2.0)
    ca = c_vals ** alpha
    CR, bR, CC, bC = 1.0, 0.0, 1.0, 0.0
    for _ in range(iters):
        JcC = lme._fourier_integral(E, alpha, CC, bC, a - 1.0, -1.0)
        CR_new = max(pre_C * JcC.real, 1e-9)
        bR_new = max(-1.0, min(1.0, cot * JcC.imag / JcC.real))
        num = 0.0 + 0.0j
        for cv_a, wv in zip(ca, c_w):
            JcR = lme._fourier_integral(E, alpha, cv_a * CR_new, bR_new,
                                        a - 1.0, -1.0)
            num += wv * cv_a * JcR
        CC_new = max(pre_C * num.real, 1e-9)
        bC_new = max(-1.0, min(1.0, cot * num.imag / num.real))
        nCR = (1 - damping) * CR + damping * CR_new
        nbR = (1 - damping) * bR + damping * bR_new
        nCC = (1 - damping) * CC + damping * CC_new
        nbC = (1 - damping) * bC + damping * bC_new
        if (abs(nCR - CR) < tol and abs(nbR - bR) < tol
                and abs(nCC - CC) < tol and abs(nbC - bC) < tol):
            CR, bR, CC, bC = nCR, nbR, nCC, nbC
            break
        CR, bR, CC, bC = nCR, nbR, nCC, nbC
    return CR, bR, CC, bC


def L_R(E, alpha, CR, bR, c_vals, c_w):
    """L_R = sum_v w_v c_v^{2 alpha} ell(E; alpha, c_v^alpha C_R, beta_R)."""
    out = 0.0 + 0.0j
    for cv, wv in zip(c_vals, c_w):
        out += wv * cv ** (2.0 * alpha) * lme.ell_plus(E, alpha,
                                                       cv ** alpha * CR, bR)
    return out


def structured_perron(E, alpha, c_vals, c_w, cb=None):
    """Perron eigenvalue of the two-leg map (eq. 7); edge at Perron = 1."""
    if cb is None:
        cb = solve_two_sided(E, alpha, c_vals, c_w)
    CR, bR, CC, bC = cb
    P = lme.ell_plus(E, alpha, CC, bC)
    Q = L_R(E, alpha, CR, bR, c_vals, c_w)
    Qc = np.conj(Q)
    K2 = (alpha / 2.0 * gammafn((1.0 - alpha) / 2.0) ** 2) ** 2
    s = sin(pi * alpha / 2.0)
    A = K2 * P * (s * s * Q + Qc)
    B = K2 * P * s * (Q + Qc)
    C = K2 * np.conj(P) * s * (Q + Qc)
    D = K2 * np.conj(P) * (s * s * Qc + Q)
    eig = np.linalg.eigvals(np.array([[A, B], [C, D]], dtype=complex))
    return float(np.abs(eig[np.argmax(np.abs(eig))])), cb


def structured_edge(alpha, c_vals, c_w, Es, verbose=True):
    """Scan E; return E where the Perron eigenvalue crosses 1 (deloc -> loc)."""
    lams = []
    for E in Es:
        lam, cb = structured_perron(float(E), alpha, c_vals, c_w)
        lams.append(lam)
        if verbose:
            print(f"  E={E:6.3f}  C_R={cb[0]:.3f} C_C={cb[2]:.3f}  "
                  f"Lambda_max={lam:.4f}", flush=True)
    lams = np.array(lams)
    return [round(0.5 * (Es[i] + Es[i + 1]), 3) for i in range(len(lams) - 1)
            if (lams[i] - 1) > 0 >= (lams[i + 1] - 1)]


def reduction_gate():
    """Constant profile, alpha=0.5: structured edge must match levy_mobility_edge."""
    log = []
    n = 20
    _, w = _gl01(n)
    c = np.ones(n)
    with Timer("structured gate alpha=0.5", log):
        edges = structured_edge(0.5, c, w, np.arange(2.5, 5.01, 0.25))
    ref = lme.find_mobility_edge(0.5, E_grid=np.arange(2.5, 5.01, 0.5),
                                 verbose=False)
    print(f"\nstructured edge (Perron=1): {edges}")
    print(f"levy_mobility_edge edge:    {ref}   [must match ~3.29]")


# ---------------------------------------------------------------------------
# Part 2: Tarquini imaginary-part stability by complex cavity population dynamics.
# ---------------------------------------------------------------------------


def cavity_typ_imG(E, alpha, eta, P=10000, K=100, iters=130, burn=75,
                   a_row=None, seed=0):
    """Typical Im G of the complex cavity RDE (LePage sum); md sec. 3.7.

    a_row None -> unstructured (single pool); else a length-P quenched row
    profile (bipartite: row pool carries the profile, column pool does not).
    Init Im G > 0 and enforce the upper half plane (the physical branch).
    """
    rng = np.random.default_rng(seed)
    tw = -2.0 / alpha
    z = E + 1j * eta
    if a_row is None:
        G = np.full(P, 0.5j, dtype=complex)
        logs = []
        for it in range(iters):
            amp = np.cumsum(rng.exponential(1.0, (P, K)), axis=1) ** tw
            Gnb = G[rng.integers(0, P, (P, K))]
            G = 1.0 / (z - (amp * Gnb).sum(axis=1))
            G.imag[:] = np.abs(G.imag)
            if it >= burn:
                logs.append(np.mean(np.log(np.clip(G.imag, 1e-300, None))))
        return float(np.exp(np.mean(logs)))
    a2 = a_row ** 2
    Gr = np.full(P, 0.5j, dtype=complex)
    Gc = np.full(P, 0.5j, dtype=complex)
    logs = []
    for it in range(iters):
        amp = np.cumsum(rng.exponential(1.0, (P, K)), axis=1) ** tw
        Gnb = Gc[rng.integers(0, P, (P, K))]
        Gr = 1.0 / (z - a2 * (amp * Gnb).sum(axis=1))
        Gr.imag[:] = np.abs(Gr.imag)
        amp = np.cumsum(rng.exponential(1.0, (P, K)), axis=1) ** tw
        nb = rng.integers(0, P, (P, K))
        Gc = 1.0 / (z - (amp * a2[nb] * Gr[nb]).sum(axis=1))
        Gc.imag[:] = np.abs(Gc.imag)
        if it >= burn:
            logs.append(np.mean(np.log(np.clip(Gr.imag, 1e-300, None))))
    return float(np.exp(np.mean(logs)))


def eta_exponent(E, alpha, a_row=None, eta_hi=1e-2, eta_lo=1e-3, **kw):
    """p = d log(Im G_typ)/d log eta : 0 delocalised, 1 localised."""
    thi = cavity_typ_imG(E, alpha, eta_hi, a_row=a_row, seed=0, **kw)
    tlo = cavity_typ_imG(E, alpha, eta_lo, a_row=a_row, seed=1, **kw)
    return log(thi / tlo) / log(eta_hi / eta_lo), thi, tlo


def cavity_localization(alpha=1.5, sigmas=(0.0, 1.5),
                        Es=(0.5, 1.5, 2.5, 3.5, 4.5, 5.5), P=10000, K=100,
                        seed=7):
    """Unstructured (delocalised baseline) vs saturating tanh row profile."""
    rng = np.random.default_rng(seed)
    for sig in sigmas:
        if sig == 0.0:
            a_row, sat = None, 0.0
        else:
            h = sig * symmetric_stable(alpha, P, rng)
            a_row = 1.0 / np.cosh(h) ** 2
            sat = float((a_row < 0.05).mean())
        print(f"\nalpha={alpha}  sigma_h={sig}  saturated-frac={sat:.2f}")
        with Timer(f"cavity sigma_h={sig}"):
            ps = []
            for E in Es:
                p, thi, tlo = eta_exponent(E, alpha, a_row=a_row, P=P, K=K)
                ps.append(p)
                print(f"  E={E:4.1f}  ImG_typ={thi:.2e}/{tlo:.2e}  p={p:+.3f}"
                      f"  {'LOCALISED' if p > 0.5 else ''}", flush=True)
        edges = [round(0.5 * (Es[i] + Es[i + 1]), 2)
                 for i in range(len(ps) - 1)
                 if (ps[i] - 0.5) < 0 <= (ps[i + 1] - 0.5)]
        print(f"  -> sigma_h={sig}: mobility edge (p=1/2) at {edges}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    if mode == "cavity":
        cavity_localization()
    else:
        reduction_gate()
