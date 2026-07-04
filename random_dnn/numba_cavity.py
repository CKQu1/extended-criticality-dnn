"""Compiled (numba) population-dynamics cavity for the heavy-tailed Jacobian
singular-value density -- the production localization-edge path.

Implements the EXACT Gauss-Seidel semantics of RMT.cavity_svd_resolvent (same
i % pop index scan, same chi^2-inside-g1 / chi^2-outside-g2 bipartite
asymmetry, same tile warm-start, pop^2 steps per doubling), with the CMS
(Chambers-Mallows-Stuck, Weron form) stable sampler inlined into the JIT
kernel. Differences from the torch path, both deliberate:

  * complex128 throughout. The torch production cavity (cfloat) prematurely
    rounds marginal Im G to zero and under-localizes, biasing the edge s_c low
    by ~0.3-0.4 at (alpha=1.5, sigma_W=1); complex128 matches the empirical
    density into the transition (validated against a complex128 torch mirror
    and N=2500 empirical SVDs on a log grid).
  * serial scan, parallelism over SEEDS at the process level
    (numba_sv_density_multiseed). Per-step parallelism or torch-style op
    dispatch is overhead-dominated for this sequential update.

chi disorder: chi = sigma_W * phi'(q^(1/alpha) z) with phi = tanh, so
chi^2 = sigma_W^2 sech^4(q^(1/alpha) z), z ~ stable(alpha, scale=2^(-1/alpha)).
q* comes from RMT.q_star_MC (imported lazily; this module itself is
torch-free).

CLI follows the RMT.py convention:
    python numba_cavity.py sv_density_log_grid alpha 1.5 float sigma_W 1.0 float
"""
import math
import numpy as np
from numba import njit

SMIN, SMAX, NBIN = 1e-2, 2e2, 64


def log_grid(smin=SMIN, smax=SMAX, nbin=NBIN):
    """Universal log-spaced grid: (edges (nbin+1,), geometric centers (nbin,))."""
    edges = np.logspace(np.log10(smin), np.log10(smax), nbin + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


@njit(cache=True, inline="always")
def _fill_sq_stable(out, alpha, scale):
    """out[k] = (scale * symmetric-alpha-stable)^2, CMS/Weron form."""
    inv_a = 1.0 / alpha
    p = (1.0 - alpha) / alpha
    for k in range(out.shape[0]):
        u = (np.random.random() - 0.5) * math.pi
        r = np.random.random()
        if r < 1e-300:
            r = 1e-300
        w = -math.log(r)
        z = (math.sin(alpha * u) / math.cos(u) ** inv_a) * (math.cos((1.0 - alpha) * u) / w) ** p
        sz = scale * z
        out[k] = sz * sz


@njit(cache=True)
def _gen_chi2(alpha, sigma_W, q, num_chis, pop):
    """chi^2 with chi = sigma_W * sech^2(q^(1/alpha) z), z ~ stable(alpha, 2^(-1/alpha))."""
    chi2 = np.empty((num_chis, pop))
    qa = q ** (1.0 / alpha)
    sc = 2.0 ** (-1.0 / alpha)
    inv_a = 1.0 / alpha
    p = (1.0 - alpha) / alpha
    for c in range(num_chis):
        for j in range(pop):
            u = (np.random.random() - 0.5) * math.pi
            r = np.random.random()
            if r < 1e-300:
                r = 1e-300
            w = -math.log(r)
            zst = (math.sin(alpha * u) / math.cos(u) ** inv_a) * (math.cos((1.0 - alpha) * u) / w) ** p
            sech2 = 1.0 / math.cosh(qa * sc * zst) ** 2
            ch = sigma_W * sech2
            chi2[c, j] = ch * ch
    return chi2


@njit(cache=True)
def _run_steps(g1, g2, chi2, z, alpha, scale, num_steps):
    L, C, P = g1.shape
    ssg2 = np.empty(P)
    ssg1 = np.empty(P)
    for i in range(num_steps):
        idx = i % P
        _fill_sq_stable(ssg2, alpha, scale)
        for s in range(L):
            zs = z[s]
            for c in range(C):
                acc = 0.0 + 0.0j
                for j in range(P):
                    acc += ssg2[j] * chi2[c, j] * g2[s, c, j]
                g1[s, c, idx] = -1.0 / (zs + acc)
        _fill_sq_stable(ssg1, alpha, scale)
        for s in range(L):
            zs = z[s]
            for c in range(C):
                acc = 0.0 + 0.0j
                for j in range(P):
                    acc += ssg1[j] * g1[s, c, j]
                g2[s, c, idx] = -1.0 / (zs + chi2[c, idx] * acc)


@njit(cache=True)
def _full_run(z, alpha, sigma_W, q, num_doublings, num_chis, seed):
    np.random.seed(seed)
    L = z.shape[0]
    pop = 2
    g1 = np.empty((L, num_chis, pop), dtype=np.complex128)
    g2 = np.empty((L, num_chis, pop), dtype=np.complex128)
    for s in range(L):
        for c in range(num_chis):
            for j in range(pop):
                g1[s, c, j] = np.random.random() + 1j * np.random.random()
                g2[s, c, j] = np.random.random() + 1j * np.random.random()
    chi2 = _gen_chi2(alpha, sigma_W, q, num_chis, pop)
    _run_steps(g1, g2, chi2, z, alpha, (2.0 * pop) ** (-1.0 / alpha), pop * pop)
    for d in range(1, num_doublings):
        newpop = pop * 2
        ng1 = np.empty((L, num_chis, newpop), dtype=np.complex128)
        ng2 = np.empty((L, num_chis, newpop), dtype=np.complex128)
        for s in range(L):
            for c in range(num_chis):
                for j in range(pop):
                    ng1[s, c, j] = g1[s, c, j]; ng1[s, c, pop + j] = g1[s, c, j]
                    ng2[s, c, j] = g2[s, c, j]; ng2[s, c, pop + j] = g2[s, c, j]
        g1, g2 = ng1, ng2
        chi2 = _gen_chi2(alpha, sigma_W, q, num_chis, newpop)
        _run_steps(g1, g2, chi2, z, alpha, (2.0 * newpop) ** (-1.0 / alpha), newpop * newpop)
        pop = newpop
    dens = np.zeros(L)
    for s in range(L):
        acc = 0.0
        for c in range(num_chis):
            for j in range(pop):
                acc += g1[s, c, j].imag + g2[s, c, j].imag
        dens[s] = acc / (math.pi * pop * num_chis)
    return dens


def numba_sv_density(sing_vals, alpha, sigma_W, q, num_doublings=9, num_chis=1, seed=0):
    """Cavity singular-value density at sing_vals; ~67 s/seed at nd=9 (1 core),
    cost x8 per extra doubling."""
    z = np.ascontiguousarray(sing_vals, dtype=np.float64)
    return _full_run(z, float(alpha), float(sigma_W), float(q),
                     int(num_doublings), int(num_chis), int(seed))


def _worker(args):
    import os
    os.environ["NUMBA_NUM_THREADS"] = "1"
    return numba_sv_density(*args)


def numba_sv_density_multiseed(sing_vals, alpha, sigma_W, q, num_doublings=9,
                               num_chis=1, seeds=range(100), workers=8):
    """Mean density over independent seeds, one serial numba scan per process.
    Returns (mean_density, per_seed_densities)."""
    from concurrent.futures import ProcessPoolExecutor
    args = [(sing_vals, alpha, sigma_W, q, num_doublings, num_chis, int(s)) for s in seeds]
    with ProcessPoolExecutor(workers) as ex:
        res = np.array(list(ex.map(_worker, args)))
    return res.mean(0), res


def sv_density_log_grid(alpha, sigma_W, q=None, num_doublings=9, num_chis=1, seed=0):
    """Density on the universal log grid; computes q* via RMT.q_star_MC if not
    given (the only torch dependency, imported lazily with a fixed seed so all
    seeds of a cell share one q)."""
    if q is None:
        import torch
        import RMT
        torch.manual_seed(0)
        q = float(RMT.q_star_MC(alpha, sigma_W)[-1])
    _, centers = log_grid()
    return numba_sv_density(centers, alpha, sigma_W, q,
                            num_doublings=num_doublings, num_chis=num_chis, seed=seed)


if __name__ == "__main__":
    # python numba_cavity.py func_name arg1_name arg1 arg1_type ...
    import sys
    from time import time

    tic = time()
    func = eval(sys.argv[1])
    args = [sys.argv[i: i + 3] for i in range(2, len(sys.argv), 3)]
    arg_dict = {arg[0]: eval(arg[2])(arg[1]) for arg in args}
    result = func(**arg_dict)
    if result is not None:
        print(result)
    toc = time()
    print(f"Script time: {toc - tic:.2f} sec")
