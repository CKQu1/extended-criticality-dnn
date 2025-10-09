import numpy as np
from tqdm.auto import tqdm

import torch
from torch.autograd.functional import jacobian
from torchlevy import stable_dist

from functools import partial
from pathlib import Path
from time import time

# Helper functions


def stable_dist_sample(*args, **kwargs):
    """Wrapper for `stable_dist.sample` that avoids nans.

    torchlevy has an issue with occasional nans, while scipy is slow at generating stable samples.
    """
    nanflag = True
    while nanflag:
        out = stable_dist.sample(*args, **kwargs)
        nanflag = out.sum().isnan()
    return out


def savetxt(fname, data):
    """Save `data` to `fname` with `np.savetxt`.

    If `data` is a dictionary, save its values to separate files suffixed by the keys.
    """
    tic = time()
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        for key, value in data.items():
            np.savetxt(f"{fname}_{key}.txt", value)
    else:
        np.savetxt(fname, data)
    print(f"Computed in {time()-tic:.2f} sec and saved to {fname}")
    return fname


# Empirical MLP functions with randomly drawn weights


def MLP_log_svdvals(
    alpha,
    sigma_W,
    sigma_b,
    phi,
    width,
    depth,
    x_init=None,
    seed=None,
    device=None,
    return_full=False,
):
    """Get the logarithm of the Jacobian singular values of a random MLP.

    Can optionally return the postactivations too.
    """
    if device is not None:
        torch.set_default_device(device)
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(width) if x_init is None else torch.as_tensor(x_init)
    postact_arr = torch.zeros((depth, width))
    log_svdvals_arr = torch.zeros((depth, width))
    for layer in range(depth):
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
        postact_arr[layer] = x
        log_svdvals_arr[layer] = torch.linalg.svdvals(jacobian(phi, h) @ M).log()
    if return_full:
        return {
            "postact": postact_arr.cpu().numpy(),
            "log_svdvals": log_svdvals_arr.cpu().numpy(),
            "svd_U": (svd := torch.linalg.svd(jacobian(phi, h) @ M))
            .U.cpu()
            .numpy(),  # cols are left sing vecs
            "svd_Vh": svd.Vh.cpu().numpy(),  # rows are right sing vecs
        }
    else:
        return log_svdvals_arr.cpu().numpy()


# MFT functions


def q_star_MC(
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    q_init=3.0,
    tol=1e-9,
    seed=None,
    num_samples=int(1e6),
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


def MFT_average(
    fn,
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    num_samples=int(1e6),
    qs=None,
):
    """Return the mean-field average of `fn` over the post-activations."""
    if alpha != 2:
        stable_samples = stable_dist_sample(
            alpha, scale=2 ** (-1 / alpha), size=num_samples
        )
    else:
        stable_samples = torch.randn(num_samples)
    if qs is None:
        qs = torch.tensor(
            q_star_MC(
                alpha,
                sigma_W,
                sigma_b,
                phi,
                num_samples=num_samples,
            ),
            device=stable_samples.device,
        )
    try:
        return fn(phi(qs[:, None] ** (1 / alpha) * stable_samples)).mean(1).tolist()
    except TypeError:  # fn is a list of functions
        fn_list = fn
        return [
            fn(phi(qs[:, None] ** (1 / alpha) * stable_samples)).mean(1).tolist()
            for fn in fn_list
        ]


# RMT functions


def resolvent_pdf(g1, g2):
    """Return the singular value pdf from a bipartised resolvent."""
    return (g1.imag.sum(1) + g2.imag.sum(1)) / (torch.pi * g2.shape[1])


def cavity_svd_resolvent(
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


def jac_cavity_svd_log_pdf(
    sing_vals: np.ndarray,
    alpha: float,
    sigma_W: float,
    sigma_b: float = 0,
    phi=torch.tanh,
    num_doublings=4,
    num_steps_fn=lambda pop_size: pop_size**2,
    progress=False,
):
    """Compute the RMT singular value distribution for the fixed point Jacobian of random MLPs.

    Because of the two sources of randomness we quickly build the resolvent by iteratively doubling
    its size and running for a number of steps scaling as the square of the population size.

    Accepts and returns a numpy array, but runs on the default torch device.
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
        g1, g2 = cavity_svd_resolvent(
            torch.tensor(sing_vals),
            alpha,
            chi_samples,
            num_steps_fn(pop_size),
            g1=None if i == 0 else g1.repeat(1, 2),
            g2=None if i == 0 else g2.repeat(1, 2),
            progress=progress,
        )
    return resolvent_pdf(g1, g2).abs().log().cpu().numpy()


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
