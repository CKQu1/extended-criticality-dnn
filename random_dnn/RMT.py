import numpy as np
from tqdm.auto import tqdm

import torch

# from torch.autograd.functional import jacobian
from torch.func import vmap, grad
from torchlevy import stable_dist

from collections import defaultdict
from functools import partial
from pathlib import Path
from time import time

# Helper functions


def stable_dist_sample(*args, **kwargs):
    """Wrapper for `stable_dist.sample` that avoids nans.

    torchlevy has an issue with occasional nans, while scipy is slow at generating stable samples.
    """
    out = stable_dist.sample(*args, **kwargs)
    nan_mask = out.isnan()
    nan_count = nan_mask.sum().item()
    while nan_count > 0:
        kwargs["size"] = nan_count
        out[nan_mask] = stable_dist.sample(*args, **kwargs)
        nan_mask = out.isnan()
        nan_count = nan_mask.sum().item()
    return out


def savetxt(path, data):
    """Save `data` to `path`.

    If the output is a dictionary, save its values to separate files with keys added to the stems.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        for key, value in data.items():
            fname = path.with_stem(f"{path.stem};{key}")
            np.savetxt(fname, value)
            print(fname)
    else:
        np.savetxt(path, data)
        print(path)


def call_save(path, func, *args, **kwargs):
    """Call `func` with `args` and `kwargs`, saving the output to `path`."""
    tic = time()
    savetxt(path, func(*args, **kwargs))
    print(f"Function {func.__name__} took {time() - tic:.2f} sec")


# Empirical MLP functions with randomly drawn weights


def MLP(
    x0,  # shape: (num_inputs, width)
    depth,
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    seed=None,
    device=None,
    compute_uv=False,  # whether to compute singular vectors as well as values
    fast=False,  # if True, do not compute jacobians
    usetqdm=True,
):
    """full stats of an MLP over depth and inputs (single realisation)

    Returns a dict of arrays with shapes:
    - postact: (depth, num_inputs, width)
    - log_svdvals: (depth, num_inputs, width)
    - svd_left, svd_right: (depth, num_inputs, width, width) if compute_uv

    If a single input is passed the num_inputs dimension is squeezed out.
    """
    if device is not None:
        torch.set_default_device(device)
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.as_tensor(x0)
    squeezed = x.ndim == 1
    if squeezed:
        x = x.unsqueeze(0)
    num_inputs, width = x.shape
    h = torch.zeros_like(x)
    diag_phi_prime = vmap(vmap(grad(phi)))
    stats = {
        k: torch.zeros(depth, num_inputs, width)
        for k in ["postact"]
        + ([] if fast else ["prejac_log_svdvals", "postjac_log_svdvals"])
    }
    if compute_uv:
        # sing vecs are in the last axis, i.e. the first sing vec is [0, 0, :]
        for jac_name in ["prejac", "postjac"]:
            for direction in ["left", "right"]:
                stats[f"{jac_name}_svd_{direction}"] = torch.zeros(
                    depth, num_inputs, width, width
                )
    for layer in tqdm(range(depth), disable=not usetqdm):
        W = (
            sigma_W
            * (2 * width) ** (-1 / alpha)
            * stable_dist_sample(alpha, size=(width, width))
        )
        if not fast:
            prejac = torch.einsum("ij, bj -> bij", W, diag_phi_prime(h))
        b = (
            sigma_b * 2 ** (-1 / alpha) * stable_dist_sample(alpha, size=width)
            if sigma_b > 0
            else 0
        )
        h = torch.einsum("ij, bj -> bi", W, x) + b
        x = phi(h)
        stats["postact"][layer] = x
        if not fast:
            # compute $$ D^l W^l $$ where $$ D^l := \diag\phi'(h^l) $$ (Eq. 2 of Pennington et al. 2018 AISTATS)
            postjac = torch.einsum("bi, ij -> bij", diag_phi_prime(h), W)
            for jac_name, jac in [("prejac", prejac), ("postjac", postjac)]:
                if compute_uv:
                    svd = torch.linalg.svd(jac)  # automatically broadcasts
                    stats[f"{jac_name}_log_svdvals"][layer] = svd.S.log()
                    # rows of Vh are the right sing vecs of the argument to torch.linalg.svd
                    stats[f"{jac_name}_svd_right"][layer] = svd.Vh
                    stats[f"{jac_name}_svd_left"][layer] = svd.U.transpose(-1, -2)
                else:
                    stats[f"{jac_name}_log_svdvals"][layer] = torch.linalg.svdvals(
                        jac
                    ).log()
    if squeezed:
        for stat_name in stats:
            stats[stat_name] = stats[stat_name].squeeze(1)
    return {stat_name: stat.cpu().numpy() for stat_name, stat in stats.items()}

# def MLP_circle(
    
# )


def multifractal_dim(v, q, dim=-1):
    return torch.log((torch.as_tensor(v).abs() ** (2 * q)).sum(dim)) / (
        np.log(v.shape[dim]) * (1 - q)
    )


def MLP_agg(
    x0,  # shape: (width,)
    depth,
    num_realisations,
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    seed=None,
    device=None,
    agg_postact=lambda x: {"sq_mean": (x**2).mean(-1)},
    agg_log_svdvals=lambda log_svdvals: {
        "mean": (mean := log_svdvals.mean(-1)),
        "std": (
            std := ((central_log_svdvals := (log_svdvals - mean[..., None])) ** 2)
            .mean(-1)
            .sqrt()
        ),
        "skew": (central_log_svdvals**3).mean(-1) / std**3,
        "kurt": (central_log_svdvals**4).mean(-1) / std**4,
    },
    agg_uv=lambda arr: {
        "D2_mean": (D2 := multifractal_dim(arr, 2)).mean(-1),
        "D2_std": D2.std(-1),
    },
):
    """Aggregated stats of random MLP realisations.

    Works with torch tensors internally but returns numpy arrays.

    Returns a dict of arrays with shapes (depth, ...)
    """
    if device is not None:
        torch.set_default_device(device)
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.tensor(x0)
    (width,) = x.shape
    x = x.broadcast_to((num_realisations, width))
    h = torch.zeros((num_realisations, width))
    diag_phi_prime = vmap(vmap(grad(phi)))
    observables = defaultdict(list)
    for layer in tqdm(range(depth)):
        W = (
            sigma_W
            * (2 * width) ** (-1 / alpha)
            * stable_dist_sample(alpha, 0, size=(num_realisations, width, width))
        )
        # layerwise jacobian of the preactivations (appears in the error term)
        prejac = torch.einsum("bij, bj -> bij", W, diag_phi_prime(h))
        b = (
            sigma_b
            * 2 ** (-1 / alpha)
            * stable_dist_sample(alpha, size=(num_realisations, width))
            if sigma_b > 0
            else 0
        )
        h = torch.einsum("bij, bj -> bi", W, x) + b
        x = phi(h)
        observables["postact"].append(agg_postact(x))
        # layerwise jacobian of the postactivations (output perturbations wrt input)
        postjac = torch.einsum("bi, bij -> bij", diag_phi_prime(h), W)
        for jac_name, jac in [("prejac", prejac), ("postjac", postjac)]:
            if agg_uv is not None:
                svd = torch.linalg.svd(jac)
                observables[f"{jac_name}_log_svdvals"].append(
                    agg_log_svdvals(svd.S.log())
                )
                observables[f"{jac_name}_svd_right"].append(agg_uv(svd.Vh))
                observables[f"{jac_name}_svd_left"].append(
                    agg_uv(svd.U.transpose(-1, -2))
                )
            else:
                observables[f"{jac_name}_log_svdvals"].append(
                    agg_log_svdvals(torch.linalg.svdvals(jac).log())
                )
    agg_stats = {}
    for obs_name, agg_dict_list in observables.items():
        agg_stats.update(
            {
                f"{obs_name}_{agg_name}": torch.stack(
                    [agg_dict[agg_name] for agg_dict in agg_dict_list]
                )
                .cpu()
                .numpy()
                for agg_name in agg_dict_list[0]
            }
        )
    return agg_stats


# MFT functions


def MFT_map(
    q0,  # shape: scalar or (num_inputs,)
    alpha,
    sigma_W,
    sigma_b=0,
    phi=torch.tanh,
    num_layers=1,
    num_realisations=1,
    agg_postact=lambda x: {
        "sq_mean": (x**2).mean(-1),
    },
    num_samples=int(1e6),
    usetqdm=True,
):
    """Mean-field map of the pseudolength over layers.

    Also returns aggregated statistics of the postactivations at each layer.

    Conversion between the pseudolength q and the previous layer's alpha-th postact moment:
    .. math:: q = sigma_W^alpha * E[|x|^alpha] + sigma_b^alpha
    the next layer's postactivation samples are then generated
    """
    # single realisation of Monte Carlo noise
    stable_samples = (
        stable_dist_sample(
            alpha, scale=2 ** (-1 / alpha), size=(num_realisations, num_samples)
        )
        if alpha != 2
        else torch.randn(num_realisations, num_samples)
    )
    if isinstance(q0, (float, int)):
        q0 = [q0]
    q = torch.tensor(q0)[:, None]
    observables = defaultdict(list)
    for layer in tqdm(range(num_layers), disable=not usetqdm):
        # alpha-th moment of postact = (q - sigma_b**alpha) / sigma_W**alpha
        # shape: (num_inputs, num_realisations, num_samples)
        postact_samples = phi(q[..., None] ** (1 / alpha) * stable_samples)
        observables["postact"].append(agg_postact(postact_samples))
        # Generate the pseudolength of the next layer (given by the current layer's alpha-th postact moment)
        q = (sigma_W**alpha) * (abs(postact_samples) ** alpha).mean(-1) + sigma_b**alpha
        observables["q"].append({"val": q})
    agg_stats = {}
    for obs_name, agg_dict_list in observables.items():
        agg_stats.update(
            {
                f"{obs_name}_{agg_name}": torch.stack(
                    [agg_dict[agg_name] for agg_dict in agg_dict_list]
                )
                .cpu()
                .numpy()
                for agg_name in agg_dict_list[0]
            }
        )
    return agg_stats  # shape of arrs: (num_layers, num_inputs, num_realisations, ...)


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
    stable_samples = (
        stable_dist_sample(alpha, scale=2 ** (-1 / alpha), size=num_samples)
        if alpha != 2
        else torch.randn(num_samples)
    )
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


# RMT functions


def resolvent_pdf(g1, g2):  # last dim
    """Return the singular value pdf from a bipartised resolvent."""
    return (g1.imag.sum(-1) + g2.imag.sum(-1)) / (torch.pi * g2.shape[-1])


def moving_avg(arr):  # last dim
    return (arr[..., 1:] + arr[..., :-1]) / 2


def empirical_avg(fn, bins, pdfs):  # last dim
    return torch.sum(moving_avg(fn(bins) * pdfs) * torch.diff(bins), dim=-1)


def cavity_svd_resolvent(
    sing_vals,  # 1d array
    alpha,
    chi_samples,  # shape: (num_chi_realisations, num_pop_samples)
    num_steps,
    g1=None,  # shape: (len(sing_vals), num_chi_realisations, num_pop_samples*matrix_aspect_ratio)
    g2=None,  # shape: (len(sing_vals), num_chi_realisations, num_pop_samples)
    progress=False,
):
    """Compute the singular value resolvent of the column-structured heavy-tailed random matrix.

    This is the resolvent of the bipartisation of the matrix $M$ where the diagonals are zero
    and the off-diagonals are $M$ and $M^T$.
    Uses a population dynamics algorithm to evaluate the RMT cavity equations of this resolvent.
    """
    chi_samples_sq = abs(chi_samples) ** 2
    if g1 is None:
        g1 = torch.rand((len(sing_vals), *chi_samples.shape), dtype=torch.cfloat)
    if g2 is None:
        g2 = torch.rand((len(sing_vals), *chi_samples.shape), dtype=torch.cfloat)
    scale = (g1.shape[-1] + g2.shape[-1]) ** (-1 / alpha)
    for i in (pbar := tqdm(range(num_steps), disable=not progress)):
        i_g1 = i % g1.shape[-1]
        i_g2 = i % g2.shape[-1]
        stable_sample_g2 = (
            stable_dist_sample(alpha, scale=scale, size=g2.shape[-1]) ** 2
        )
        g1[..., i_g1] = -1 / (
            sing_vals[:, None] + (stable_sample_g2 * chi_samples_sq * g2).sum(-1)
        )
        stable_sample_g1 = (
            stable_dist_sample(alpha, scale=scale, size=g1.shape[-1]) ** 2
        )
        g2[..., i_g2] = -1 / (
            sing_vals[:, None]
            + chi_samples_sq[:, i_g2] * (stable_sample_g1 * g1).sum(-1)
        )
        if progress and num_steps > 100 and i % (num_steps // 100) == 0:
            # get the estimate of the norm (should be 1)
            norms = empirical_avg(lambda x: 1, sing_vals, resolvent_pdf(g1, g2).T)
            pbar.set_postfix_str(
                f"norm estimate: {norms.mean().item():.4f}±{norms.std().item():.4f}"
            )
    return g1, g2


def jac_cavity_svd_log_pdf(
    sing_vals: np.ndarray,
    alpha: float,
    sigma_W: float,
    sigma_b: float = 0,
    phi=torch.tanh,
    q=None,
    num_doublings=4,
    num_steps_fn=lambda pop_size: pop_size**2,
    progress=False,
    seed=None,
    regulariser=None,
    num_chis=1,  # number of chi realisations
):
    """Compute the RMT singular value distribution for the fixed point Jacobian of random MLPs.

    Because of the two sources of randomness we quickly build the resolvent by iteratively doubling
    its size and running for a number of steps scaling as the square of the population size.
    If there is a regulariser, we add it at the last doubling step.

    Accepts and returns a numpy array, but runs on the default torch device.
    """
    if seed is not None:
        torch.manual_seed(seed)
    sing_vals_tensor = torch.tensor(sing_vals)
    if q is None:
        q = q_star_MC(alpha, sigma_W, sigma_b, phi)[-1]
    pop_size = 1
    phi_prime = torch.vmap(torch.vmap(torch.func.grad(phi)))
    for i in range(num_doublings):
        pop_size *= 2
        if regulariser is not None and i == num_doublings - 1:
            if regulariser is True:
                regulariser = 10 / (pop_size * resolvent_pdf(g1, g2).abs().max().item())
                if progress:
                    print(f"Using regulariser={regulariser:.2e}")
            sing_vals_tensor = sing_vals_tensor + 1j * regulariser
        if alpha != 2:
            stable_samples = stable_dist_sample(
                alpha, scale=2 ** (-1 / alpha), size=(num_chis, pop_size)
            )
        else:
            stable_samples = torch.randn(num_chis, pop_size)
        chi_samples = sigma_W * phi_prime(q ** (1 / alpha) * stable_samples)
        g1, g2 = cavity_svd_resolvent(
            sing_vals_tensor,
            alpha,
            chi_samples,
            num_steps_fn(pop_size),
            g1=None if i == 0 else g1.repeat(1, 1, 2),
            g2=None if i == 0 else g2.repeat(1, 1, 2),
            progress=progress,
        )
    return (
        resolvent_pdf(g1, g2).abs().log().cpu().numpy()
    )  # shape: (len(sing_vals), num_chi_realisations)


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
    if result is not None:
        print(result)
    toc = time()
    print(f"Script time: {toc - tic:.2f} sec")
