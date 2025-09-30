import numpy as np
import torch
from torch.autograd.functional import jacobian
from torchlevy import stable_dist

# from scipy.stats import levy_stable


def stable_dist_sample(*args, **kwargs):
    """Wrapper for `stable_dist.sample` that avoids nans"""
    nanflag = True
    while nanflag:
        # torchlevy has this issue with occasional nans
        out = stable_dist.sample(*args, **kwargs)
        nanflag = out.sum().isnan()
    return out


# LogSingValsMLP[\[Alpha]_, \[Sigma]w_, \[Sigma]b_, \[Phi]_, width_, depth_, fn_ : Identity ] :=
def MLP_log_svdvals(alpha, sigma_W, sigma_b, phi, width, depth, seed=None, device=None):
    if seed is not None:
        torch.manual_seed(seed)
    if device is not None:
        torch.set_default_device(device)
    x = torch.randn(width)
    # for _ in (pbar := tqdm(range(depth))):
    for _ in range(depth):
        # M = torch.from_numpy(levy_stable.rvs(alpha, 0, scale=sigma_W * width**(-1/alpha), size=(width, width))).to(device, dtype=x.dtype)
        M = stable_dist_sample(
            alpha, 0, scale=sigma_W * (2 * width) ** (-1 / alpha), size=(width, width)
        )
        b = (
            stable_dist_sample(alpha, 0, scale=sigma_b * 2 ** (-1 / alpha), size=width)
            if sigma_b > 0
            else 0
        )
        h = M @ x + b
        x = phi(h)
        # pbar.set_postfix({"mean": x.mean().item(), "std": x.std().item()})
    # fn@Log@SingularValueList[DiagonalMatrix[\[Phi]'[h]] . M]
    return torch.linalg.svdvals(jacobian(phi, h) @ M).log().cpu().numpy()


def worker(sigma_W, alphas, sigma_b, phi, width, depth):
    return [
        MLP_log_svdvals(alpha, sigma_W, sigma_b, phi, width, depth).mean().item()
        for alpha in alphas
    ]


from pathlib import Path
from time import time


def savetxt(fname, func, *args, **kwargs):
    tic = time()
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    data = func(*args, **kwargs)
    np.savetxt(fname, data)
    print(f"Computed in {time()-tic:.2f} sec and saved to {fname}")
    return fname


from tqdm.auto import tqdm


def resolvent_pdf(g1, g2):
    """Return the singular value pdf from a bipartised resolvent."""
    return (g1.imag.sum(1) + g2.imag.sum(1)) / (torch.pi * g2.shape[1])


def singular_value_resolvent(
    sing_vals,
    alpha,
    chi_samples,
    num_steps,
    g1=None,
    g2=None,
    progress=False,
):
    """Compute the singular value resolvent of the column-structured heavy-tailed random matrix.

    This is the resolvent of the bipartisation of the matrix $M$ where the diagonals are zero
    and the off-diagonals are $M$ and $M^T$.
    Uses a population dynamics algorithm to evaluate the RMT cavity equations of this resolvent.
    """
    chi_samples_sq = abs(chi_samples) ** 2
    if g1 is None:
        g1 = torch.rand((len(sing_vals), len(chi_samples)), dtype=torch.cfloat)
    if g2 is None:
        g2 = torch.rand((len(sing_vals), len(chi_samples)), dtype=torch.cfloat)
    scale = (g1.shape[1] + g2.shape[1]) ** (-1 / alpha)
    for i in (pbar := tqdm(range(num_steps), disable=not progress)):
        i_g1 = i % g1.shape[1]
        i_g2 = i % g2.shape[1]
        stable_sample_g2 = stable_dist_sample(alpha, scale=scale, size=g2.shape[1]) ** 2
        g1[:, i_g1] = -1 / (sing_vals + (stable_sample_g2 * chi_samples_sq * g2).sum(1))
        stable_sample_g1 = stable_dist_sample(alpha, scale=scale, size=g1.shape[1]) ** 2
        g2[:, i_g2] = -1 / (
            sing_vals + chi_samples_sq[i_g2] * (stable_sample_g1 * g1).sum(1)
        )
        if progress and num_steps > 100 and i % (num_steps // 100) == 0:
            pbar.set_postfix_str(torch.std_mean(resolvent_pdf(g1, g2)))
    return g1, g2


def singular_value_pdf(*args, **kwargs):
    return resolvent_pdf(*singular_value_resolvent(*args, **kwargs))


def q_star_MC(
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    q_init=3.0,
    tol=1e-9,
    seed=None,
    num_samples=1000000,
):
    """Monte-Carlo version of the fixed point pseudolength.

    A MC approach was needed because evaluating pdfs is slow on scipy
    and inaccurate on torchlevy, especially for alpha close to 1.
    This issue does not exist for torchlevy when generating random samples.

    Takes about 10 MB of memory and 0.2 seconds for 1 million samples on cpu,
    with a numerical error of about 0.5%.

    Returns a list of floats.
    """
    qs = [q_init]
    if seed is not None:
        torch.manual_seed(seed)
    if alpha != 2:
        stable_samples = stable_dist_sample(
            alpha, scale=2 ** (-1 / alpha), size=num_samples
        )
    else:
        stable_samples = torch.randn(num_samples)
    converged = False
    while not converged:
        q = (sigma_W**alpha) * (
            abs(phi(qs[-1] ** (1 / alpha) * stable_samples)) ** alpha
        ).mean() + sigma_b**alpha
        qs.append(q.item())
        converged = abs(qs[-1] - qs[-2]) < tol
    if seed is not None:
        torch.seed()
    return qs


def jacobian_singular_value_pdf(
    sing_vals,
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    num_doublings=4,
    num_steps_fn=lambda pop_size: pop_size**2,
    progress=False,
):
    """Compute the RMT singular value distribution for the fixed point Jacobian of random MLPs.

    Because of the two sources of randomness we quickly build the resolvent by iteratively doubling
    its size and running for a number of steps scaling as the square of the population size.
    """
    q_star = q_star_MC(alpha, sigma_W, sigma_b, phi)[-1]
    pop_size = 1
    for i in range(num_doublings):
        pop_size *= 2
        if alpha != 2:
            stable_samples = stable_dist_sample(
                alpha, scale=2 ** (-1 / alpha), size=pop_size
            )
        else:
            stable_samples = torch.randn(pop_size)
        chi_samples = sigma_W * torch.vmap(torch.func.grad(phi))(
            q_star ** (1 / alpha) * stable_samples
        )
        g1, g2 = singular_value_resolvent(
            sing_vals,
            alpha,
            chi_samples,
            num_steps_fn(pop_size),
            g1=None if i == 0 else g1.repeat(1, 2),
            g2=None if i == 0 else g2.repeat(1, 2),
            progress=progress,
        )
    return resolvent_pdf(g1, g2).cpu().numpy()


if __name__ == "__main__":
    # Example usage:
    # python filename.py func_name arg1_name arg1 arg1_type arg2_name arg2 arg2_type ...
    # the arguments will be passed in as keyword args
    import sys
    from time import time

    tic = time()
    func = eval(sys.argv[1])
    args = [sys.argv[i : i + 3] for i in range(2, len(sys.argv), 3)]
    arg_dict = {arg[0]: eval(arg[2])(arg[1]) for arg in args}
    result = func(**arg_dict)
    print(result)
    toc = time()
    print(f"Time: {toc - tic:.2f} sec")
