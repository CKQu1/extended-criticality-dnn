import torch
from torch.autograd.functional import jacobian
from torchlevy import stable_dist

# from scipy.stats import levy_stable

torch.set_default_device("cuda")


def stable_dist_sample(*args, **kwargs):
    nanflag = True
    while nanflag:
        # torchlevy has this issue with occasional nans
        out = stable_dist.sample(*args, **kwargs)
        nanflag = out.sum().isnan()
    return out


# LogSingValsMLP[\[Alpha]_, \[Sigma]w_, \[Sigma]b_, \[Phi]_, width_, depth_, fn_ : Identity ] :=
def MLP_log_svdvals(alpha, sigma_W, sigma_b, phi, width, depth):
    x = torch.randn(width)
    # for _ in (pbar := tqdm(range(depth))):
    for _ in range(depth):
        # M = torch.from_numpy(levy_stable.rvs(alpha, 0, scale=sigma_W * width**(-1/alpha), size=(width, width))).to(device, dtype=x.dtype)
        M = stable_dist_sample(
            alpha, 0, scale=sigma_W * width ** (-1 / alpha), size=(width, width)
        )
        b = sigma_b * torch.randn(width) if sigma_b > 0 else 0
        h = M @ x + b
        x = phi(h)
        # pbar.set_postfix({"mean": x.mean().item(), "std": x.std().item()})
    # fn@Log@SingularValueList[DiagonalMatrix[\[Phi]'[h]] . M]
    return torch.linalg.svdvals(jacobian(phi, h) @ M).log()


def worker(sigma_W, alphas, sigma_b, phi, width, depth):
    return [
        MLP_log_svdvals(alpha, sigma_W, sigma_b, phi, width, depth).mean().item()
        for alpha in alphas
    ]
