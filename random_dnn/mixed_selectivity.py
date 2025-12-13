import RMT
import torch
import torchvision
import numpy as np
from tqdm.auto import tqdm


def MFT_map(
    dataset_name,
    alpha,
    sigma_W,
    sigma_b=0.0,
    num_layers=10,
    chunk_size=100,
    seed=42,
    num_inputs=None,  # for debugging
):
    if dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root="fig/datasets", transform=torchvision.transforms.ToTensor()
        )
    elif dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(
            root="fig/datasets", transform=torchvision.transforms.ToTensor()
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented.")
    dataset_data = np.array(dataset.data[:num_inputs], dtype=np.float32)
    flattened_data = dataset_data.reshape(len(dataset_data), -1)
    normed_data = (
        flattened_data / np.mean(flattened_data**2, axis=1, keepdims=True) ** 0.5
    )
    fp_norm = (
        RMT.MFT_map(RMT.q_star_MC(alpha, sigma_W)[-1], alpha, sigma_W, usetqdm=False)[
            "postact_sq_mean"
        ][0]
        ** 0.5
    )
    x0 = normed_data.reshape(len(normed_data), -1) * fp_norm
    q0 = sigma_W**alpha * (abs(x0) ** alpha).mean(-1) + sigma_b**alpha
    MFT_maps = []
    for q0_chunk in tqdm(
        np.array_split(q0, max(1, len(q0) // chunk_size), axis=0),
        desc="MFT map chunks",
    ):
        torch.manual_seed(seed)
        MFT_maps.append(
            RMT.MFT_map(
                q0_chunk,
                alpha,
                sigma_W,
                sigma_b=sigma_b,
                num_layers=num_layers,
                agg_postact=lambda x: {
                    "sq_mean": (x**2).mean(-1),
                    "alpha_mean": (abs(x) ** alpha).mean(-1),
                },
                usetqdm=False,
            )
        )
    # average over Monte-Carlo realisations (axis=2), then concatenate chunks of inputs (axis=1)
    agg_stats = {
        k: np.concatenate([np.mean(mft[k], axis=2) for mft in MFT_maps], axis=1)
        for k in MFT_maps[0]
    }
    agg_stats["postact_alpha_mean"] = np.array(
        [(abs(x0) ** alpha).mean(-1), *agg_stats["postact_alpha_mean"]]
    )
    agg_stats["postact_sq_mean"] = np.array(
        [(x0**2).mean(-1), *agg_stats["postact_sq_mean"]]
    )
    return agg_stats  # shape of values: (num_layers + 1, num_inputs)