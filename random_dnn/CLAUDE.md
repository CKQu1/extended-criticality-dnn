# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

- The user does their own `git` staging and commits. Make file edits/deletions as requested (including `git rm` when retiring a file), but do not run `git add`/`git commit`, and do not surface reminders or warnings about staging or commit state.

## Project Overview

Research implementation for the paper *"Dynamical and computational properties of heavy-tailed deep neural networks"* (Qu, Wardak, Gong). The central object is a randomly-initialised MLP whose weight matrices are drawn from Lévy alpha-stable distributions, parameterised by stability index `alpha` (α ∈ [1,2]) and scale `sigma_W`. The code studies phase transitions in dynamical and computational properties across the `(alpha, sigma_W)` plane.

The primary working directory is `random_dnn/`. The parent repo `extended-criticality-dnn/` contains training, post-training analysis, and visualisation code that feeds into or draws from this subfolder's outputs.

## Setup

```bash
pip install -r ../requirements.txt   # PyTorch, NumPy, SciPy
pip install torchlevy                # needed by RMT.py
```

`RMT.py` uses `torchlevy` for GPU-accelerated Lévy sampling (wraps with a NaN-retry loop in `stable_dist_sample`). `random_dnn.py` uses `scipy.stats.levy_stable` instead and is CPU-only.

## Repository Structure

```
extended-criticality-dnn/
├── random_dnn/               ← primary working directory
│   ├── RMT.py               # core module: empirical MLP stats + RMT theory
│   ├── theory_submit.py     # cluster submission + data management utilities
│   ├── random_dnn.py        # older CV/SEM analysis (scipy, positional CLI)
│   ├── mixed_selectivity.py # mixed selectivity MFT analysis
│   ├── fig/                 # output data and figures (auto-created)
│   ├── *.ipynb              # exploratory notebooks
│   └── rsync-gadi/setonix   # sync scripts for clusters
├── UTILS/
│   ├── utils_dnn.py         # IPR, D_q, transition line loading
│   └── data_utils.py        # MNIST/CIFAR-10/FashionMNIST/Gaussian loaders
├── train_DNN_code/
│   └── model_loader.py      # named architecture registry (fc3_mnist_tanh, alexnet, …)
├── dq_analysis/             # Jacobian eigenvector and NPC analysis (post-training)
├── geometry_analysis/       # circular manifold propagation through MLPs
├── pretrained_workflow/     # fits Lévy-stable to pretrained PyTorch/TF weights
├── train_supervised.py      # main training entry point (MLP + CNN)
├── path_names.py            # defines `root_data` path (cluster-aware)
└── constants.py             # cluster configs, logging defaults
```

## Key Modules in `random_dnn/`

### `RMT.py` — core computations (PyTorch)

All functions accept `alpha` and `sigma_W` as floats (not the `alpha100`/`g100` integer convention used in older scripts).

| Function | Purpose |
|----------|---------|
| `MLP(x0, depth, alpha, sigma_W, ...)` | Single-realisation MLP: returns `postact`, `prejac_log_svdvals`, `postjac_log_svdvals` (and optionally singular vectors) |
| `MLP_agg(x0, depth, num_realisations, alpha, sigma_W, ...)` | Ensemble-averaged MLP stats with configurable aggregation lambdas |
| `MFT_map(q0, alpha, sigma_W, ...)` | Mean-field theory pseudolength map over layers |
| `q_star_MC(alpha, sigma_W, ...)` | Monte Carlo fixed-point pseudolength (iterates until convergence) |
| `cavity_svd_resolvent(...)` | Population dynamics algorithm for the RMT cavity equations |
| `jac_cavity_svd_log_pdf(sing_vals, alpha, sigma_W, ...)` | RMT singular value log-pdf for the fixed-point Jacobian |
| `multifractal_dim(v, q)` | D_q multifractal dimension from a vector of weights |
| `stable_dist_sample(alpha, ...)` | NaN-safe wrapper around `torchlevy.stable_dist.sample` |

`RMT.py` CLI uses **keyword args with explicit type conversion**:
```bash
python RMT.py func_name arg_name arg_val arg_type [arg_name arg_val arg_type ...]
# e.g.:
python RMT.py q_star_MC alpha 1.5 float sigma_W 1.0 float
```

### `theory_submit.py` — cluster submission & data management

**Submission functions** (auto-detect cluster from `platform.node()`):

| Function | Purpose |
|----------|---------|
| `submit_jac_cavity_svd_log_pdf(...)` | Submit `RMT.jac_cavity_svd_log_pdf` grid over `(alpha100, sigmaW100)` |
| `submit_MLP_agg(...)` | Submit `RMT.MLP_agg` grid |
| `submit_mixed_selectivity(...)` | Submit `mixed_selectivity.MFT_map` grid |
| `submit_python_funcs(func_calls_dict, dir, ...)` | General-purpose cluster workflow; skips already-computed files |

**Data management**:

| Function | Purpose |
|----------|---------|
| `consolidate_arrays(path, pattern, ...)` | Merges `*.txt` result files into a `.npz` archive (parallel, incremental) |
| `updatez(file, **kwds)` | Appends new keys to an existing `.npz` without duplicating |
| `savetxt(path, data)` | Saves array or dict of arrays; creates parent dirs |
| `call_save(path, func, *args, **kwargs)` | Calls `func` and saves result via `savetxt` |

`theory_submit.py` CLI uses the same keyword-arg convention as `RMT.py`:
```bash
python theory_submit.py func_name arg_name arg_val arg_type ...
```

**Cluster environments** (auto-detected):
- `physics.usyd.edu.au`: PBS arrays, venv `/import/silo3/wardak/.venv`, jobs `/taiji1/wardak/job`, queue `defaultQ`
- `gadi.nci.org.au`: PBS single-job (no arrays), project `au05`, queues `normal/normalsr/normalbw/normalsl`
- `setonix*.pawsey`: SLURM arrays, project `pawsey1267`, queue `work`

Useful cluster commands (from `theory_submit.py` docstring):
```bash
qstat | tr -s ' ' | cut -d' ' -f5 | sort | uniq -c          # job array status
tail -qn1 *OU | cut -d' ' -f4 | python -c "import numpy as np, sys; arr = np.loadtxt(sys.stdin)/60/60; print(np.mean(arr), np.std(arr), np.min(arr), np.max(arr), 'hours')"
nci_account   # NCI quota
```

### `random_dnn.py` — older CV/SEM analysis (SciPy, CPU)

Uses positional CLI: `python random_dnn.py FUNCTION_NAME ARG1 ... ARGN`

```bash
python random_dnn.py SEM_save N L N_theta alpha100 g100 rep   # save one rep
python random_dnn.py SEM_preplot alpha100 g100                 # average over reps
python random_dnn.py SEM_plot path 15,25,35                    # phase transition plot
```

Note: uses `alpha100`/`g100` integer convention (multiply actual value by 100).

## Output File Conventions

Results are saved under `fig/` with structured subdirectory names:
```
fig/jac_cavity_svd_log_pdf/num_doublings=8;logspace_params=...;num_chis=1/
    alpha100=150;sigmaW100=100;seed=0.txt
fig/MLP_agg/width=1000;depth=50;num_realisations=50/
    alpha100=150;sigmaW100=100;seed=0;log_svdvals_mean.txt
```

After collection, `consolidate_arrays()` merges these into a `.npz` beside the directory and deletes the individual `.txt` files.

## Parent Repo Workflows

**Training** (run from repo root):
```bash
python train_supervised.py train_ht_dnn mnist 100 100 sgd 1024 None None {root_path} 0.001 650
python train_supervised.py train_ht_cnn cifar10 100 100 sgd alexnet fc_default
```

**Accuracy phase transition plots**:
```bash
python tranasition_plot_functions/mlp_acc_phase.py mlp_accloss_phase test
python tranasition_plot_functions/cnn_acc_phase.py cnn_accloss_phase
```

**Jacobian / NPC post-training analysis** (`dq_analysis/`):
```bash
python dq_analysis/jac_fcn.py jac_save post alpha100 g100 input_idxs epoch
python dq_analysis/jac_fcn.py jac_to_dq alpha100 g100 input_idx epoch post reig
python dq_analysis/npc_fcn.py npc_layerwise post alpha100 g100 [0,650]
python dq_analysis/npc_fcn.py npc_layerwise_d2 post alpha100 g100 [0,650]
```

**Pretrained network weight fitting** (`pretrained_workflow/`):
```bash
python pretrained_workflow/pretrained_wfit.py pretrained_allfit /path/to/weights 0 True
python pretrained_workflow/sep_new_fig1.py
```

**Geometry** (`geometry_analysis/`):
```bash
python geometry_analysis/great_circle_proj2.py gcircle_save N L N_thetas 100alpha 100sigma_w
python geometry_analysis/great_circle_proj2.py gcircle_plot
```

## Parameter Conventions

- **`RMT.py` / `theory_submit.py`**: use actual floats (`alpha=1.5`, `sigma_W=1.0`)
- **`random_dnn.py` and older scripts**: use integers scaled by 100 (`alpha100=150`, `g100=100`)
- `path_names.root_data` provides the cluster-aware data root; set or check `path_names.py` when running locally
