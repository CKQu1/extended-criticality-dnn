import os
import random
from os import makedirs, remove
from os.path import isdir, isfile, join

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # set before importing torch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import levy_stable
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


def get_weight(shape, alpha, g, seed):
    # Matches tf_models.get_weight(): N_eff = k_h * k_w * in_channels
    n_eff = int(shape[0] * shape[1]) * shape[2]
    scale = g * (0.5 / n_eff) ** (1.0 / alpha)
    weight = levy_stable.rvs(alpha, 0, size=shape, scale=scale, random_state=seed)
    return torch.tensor(weight, dtype=torch.float32)


def get_uniform_weight(shape):
    # Matches tf_models.get_uniform_weight(): N_in = shape[0]
    n_in = shape[0]
    limit = 1.0 / np.sqrt(n_in)
    return torch.empty(shape, dtype=torch.float32).uniform_(-limit, limit)


def circular_padding(input_, width, kernel_size):
    """Exact PyTorch translation of tf_models.circular_padding()."""
    begin = kernel_size // 2
    end = kernel_size - 1 - begin

    if begin > 0:
        tmp_up = input_[:, :, width - begin:width, :]
    else:
        tmp_up = input_[:, :, :0, :]

    if end > 0:
        tmp_down = input_[:, :, 0:end, :]
    else:
        tmp_down = input_[:, :, :0, :]

    tmp = torch.cat([tmp_up, input_, tmp_down], dim=2)
    new_width = width + kernel_size - 1

    if begin > 0:
        tmp_left = tmp[:, :, :, width - begin:width]
    else:
        tmp_left = tmp[:, :, :, :0]

    if end > 0:
        tmp_right = tmp[:, :, :, 0:end]
    else:
        tmp_right = tmp[:, :, :, :0]

    padded = torch.cat([tmp_left, tmp, tmp_right], dim=3)

    expected_hw = (new_width, new_width)
    if padded.shape[-2:] != expected_hw:
        raise RuntimeError(
            f"Circular padding produced shape {padded.shape[-2:]}, expected {expected_hw}."
        )
    return padded


def tf_same_padding(kernel_size, stride=1):
    if stride != 1:
        raise ValueError("This helper is only used for stride=1 SAME padding.")
    return kernel_size // 2


def conv2d_tf_same(input_, weight, stride):
    """
    Emulate TensorFlow's tf.nn.conv2d(..., padding='SAME') in PyTorch.

    For stride > 1, TensorFlow may use asymmetric padding, while
    F.conv2d(..., padding=kernel_size//2) is always symmetric.
    """
    kernel_h, kernel_w = weight.shape[-2:]
    input_h, input_w = input_.shape[-2:]

    out_h = int(np.ceil(input_h / stride))
    out_w = int(np.ceil(input_w / stride))

    pad_h = max((out_h - 1) * stride + kernel_h - input_h, 0)
    pad_w = max((out_w - 1) * stride + kernel_w - input_w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        input_ = F.pad(input_, (pad_left, pad_right, pad_top, pad_bottom))

    return F.conv2d(input_, weight, stride=stride, padding=0)


class ConvModelTorch(nn.Module):
    def __init__(self, alpha, g, seed, depth, c_size, k_size):
        super().__init__()
        self.alpha = alpha
        self.g = g
        self.seed = seed
        self.depth = depth
        self.c_size = c_size
        self.k_size = k_size

        self.phi = torch.tanh
        self.conv_layers = nn.ParameterList()

        np.random.seed(self.seed)
        levy_stable_seeds = np.random.randint(1000000, size=depth)

        init_val = get_weight([k_size, k_size, 1, c_size], alpha, g, levy_stable_seeds[0])
        self.conv_layers.append(nn.Parameter(init_val.permute(3, 2, 0, 1).contiguous()))

        shape = [k_size, k_size, c_size, c_size]
        for j in range(depth - 1):
            init_val = get_weight(shape, alpha, g, levy_stable_seeds[j + 1])
            self.conv_layers.append(nn.Parameter(init_val.permute(3, 2, 0, 1).contiguous()))

        init_val = get_uniform_weight([c_size, 10])
        self.logit_W = nn.Parameter(init_val)

    def forward(self, inputs):
        z = inputs.reshape(-1, 1, 28, 28)
        new_width = 7

        for lidx, layer in enumerate(self.conv_layers):
            if lidx == 0:
                z = conv2d_tf_same(z, layer, stride=1)
            elif lidx in [1, 2]:
                z = conv2d_tf_same(z, layer, stride=2)
            else:
                z_pad = circular_padding(z, new_width, self.k_size)
                z = F.conv2d(z_pad, layer, stride=1, padding=0)

            z = self.phi(z)

        z_ave = z.mean(dim=(2, 3))
        return z_ave @ self.logit_W


# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     if os.environ.get("TF_TRAIN_DETERMINISTIC", "1") == "1":
#         # torch.use_deterministic_algorithms(True)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_mnist(data_root):
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True)

    x_train = torch.tensor(train_dataset.data.numpy(), dtype=torch.float32) / 255.0
    y_train = torch.tensor(train_dataset.targets.numpy(), dtype=torch.long)
    x_test = torch.tensor(test_dataset.data.numpy(), dtype=torch.float32) / 255.0
    y_test = torch.tensor(test_dataset.targets.numpy(), dtype=torch.long)
    return (x_train, y_train), (x_test, y_test)


def build_dataloaders(x_train, y_train, x_test, y_test, bs, seed):
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
    )
    return train_loader, test_loader


def evaluate(loader, model, loss_fn, device):
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def run_model(alpha100, g100, seed, depth, c_size, k_size, epochs, lr, momentum, bs, root_path):
    print("Settings")
    print(f"c_size = {c_size}, k_size = {k_size}, lr = {lr}, mom = {momentum}, bs = {bs} \n")
    print(f"alpha100 = {alpha100}, g100 = {g100}, seed = {seed}, depth = {depth}, epochs = {epochs} \n")

    alpha100, g100, seed, depth = int(alpha100), int(g100), int(seed), int(depth)
    c_size, k_size = int(c_size), int(k_size)
    epochs, lr, momentum, bs = int(epochs), float(lr), float(momentum), int(bs)

    set_seed(seed)

    # data_root = join(os.getcwd(), "data")
    data_root = join(os.getcwd(), ".droot", "data")
    if not isdir(data_root):
        makedirs(data_root)

    (x_train, y_train), (x_test, y_test) = load_mnist(data_root)
    train_loader, test_loader = build_dataloaders(x_train, y_train, x_test, y_test, bs, seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on GPU" if device.type == "cuda" else "Running on CPU")
    print("")

    model = ConvModelTorch(
        alpha=alpha100 / 100,
        g=g100 / 100,
        seed=seed,
        depth=depth,
        c_size=c_size,
        k_size=k_size,
    ).to(device)

    probe_batch = max(2, min(bs, 8))
    try:
        _ = model(torch.zeros((probe_batch, 28, 28), dtype=torch.float32, device=device))
    except RuntimeError as exc:
        if device.type != "cuda":
            raise
        print(f"GPU probe failed on {device}: {exc}")
        print("Falling back to CPU because the visible GPU stack is unusable for Conv2D.")
        device = torch.device("cpu")
        model = ConvModelTorch(
            alpha=alpha100 / 100,
            g=g100 / 100,
            seed=seed,
            depth=depth,
            c_size=c_size,
            k_size=k_size,
        ).to(device)
        _ = model(torch.zeros((probe_batch, 28, 28), dtype=torch.float32, device=device))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    metric_cols = ["train loss", "train acc", "test loss", "test acc"]
    metrics_ls = []
    save_dir = join(root_path, f"cnn{depth}_{alpha100}_{g100}_{seed}")
    if not isdir(save_dir):
        makedirs(save_dir)

    for epoch in tqdm(range(epochs)):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(train_loader, model, loss_fn, device)
        test_loss, test_accuracy = evaluate(test_loader, model, loss_fn, device)

        metrics_ls.append([train_loss, train_accuracy, test_loss, test_accuracy])
        df = pd.DataFrame(metrics_ls, columns=metric_cols)
        df.to_csv(join(save_dir, "_acc_loss"))

        print(f"Epoch {epoch + 1}:")
        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            f"                Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
        )

    if isfile(join(save_dir, "_acc_loss")):
        remove(join(save_dir, "_acc_loss"))
    df_path = join(save_dir, "acc_loss")
    df.to_csv(df_path)
    print(f"Data saved as {df_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} FUNCTION_NAME ARG1 ... ARGN")
        raise SystemExit(1)
    globals()[sys.argv[1]](*sys.argv[2:])
