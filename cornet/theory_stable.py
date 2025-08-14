import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import hypsecant
from scipy.stats import norm, levy_stable
from tqdm import tqdm

epsabs = 1e-2
epsrel = 1e-2
dz = 0.05
# zmin = -10.0
# zmax = 10.0
zmin = -50.0
zmax = 50.0

# def fast_integral(integrand, zmin, zmax, dz, ndim=1):
#     zs = np.r_[zmin:zmax:dz]
#     if ndim > 1:
#         zgrid = np.meshgrid(*((zs,) * ndim))
#     else:
#         zgrid = (zs,)
#     out = integrand(*zgrid)
#     return out.sum(tuple(np.arange(ndim))) * dz**ndim

def fast_integral(integrand, zmin, zmax, dz, ndim=1):
    zs = np.r_[zmin:zmax:dz]
    out = integrand(zs)
    return out.sum() * dz**ndim


def qmap(qin, alpha=1, g=1, nonlinearity=np.tanh,
         #epsabs=epsabs, epsrel=epsrel, zmin=-10, zmax=10, dz=dz, fast=True):
         epsabs=epsabs, epsrel=epsrel, zmin=-10, zmax=10, dz=dz, fast=True):
    qin = np.atleast_1d(qin)
    # Perform Gaussian integral
    def integrand(z):
        #return levy_stable.pdf(z[:, None], alpha=alpha, beta=0, loc=0, scale=1) * nonlinearity(qin[None, :]/2 * z[:, None])**alpha
        return levy_stable.pdf(z[:, None], alpha=alpha, beta=0, loc=0, scale=1) * np.abs(nonlinearity(qin/2 * z[:, None]))**alpha
    integral = fast_integral(integrand, zmin, zmax, dz=dz)
    return g**alpha * integral

# test
"""
def integrand(z):
    qin = 3
    #return levy_stable.pdf(z[:, None], alpha=1, beta=0, loc=0, scale=1) * np.tanh(qin[None, :]/2 * z[:, None])**alpha
    return levy_stable.pdf(z[:, None], alpha=1, beta=0, loc=0, scale=1) * np.tanh(qin/2 * z[:, None])**alpha
"""

def q_fixed_point(alpha=1, g=1, nonlinearity=np.tanh, max_iter=500, tol=1e-9, qinit=3.0, fast=True, tol_frac=0.01):
    """Compute fixed point of q map"""
    global qs, q, qnew, t, err
    q = qinit
    qs = []
    for i in tqdm(range(max_iter)):
        qnew = qmap(q, alpha, g, nonlinearity, fast=fast)
        err = np.abs(qnew - q)    
        if isinstance(q,np.ndarray):
            qs.append(q.item())
        else:
            qs.append(q)
        if err < tol:
            break
        q = qnew
    # Find first time it gets within tol_frac fracitonal error of q*
    frac_err = (np.array(qs) - q)**2 / (1e-9 + q**2)
    t = np.flatnonzero(frac_err < tol_frac)[0]
    return t, q

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    globals()[sys.argv[1]](*sys.argv[2:])    