import numpy as np
import torch
from torch.autograd.functional import jacobian
from torchlevy import stable_dist

# from scipy.stats import levy_stable


def stable_dist_sample(*args, **kwargs):
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
