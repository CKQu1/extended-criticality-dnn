# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

- Do not run any `git` command unless the user has explicitly approved it. Immediately before running a `git` command, ask for confirmation for that specific command. The user does their own `git` staging and commits; make file edits/deletions as requested (including `git rm` when retiring a file), but do not run `git` commands speculatively or as routine cleanup.
- When drafting commit messages for the user, do not include a `Co-authored-by` trailer unless the user explicitly asks for it.
- Source files (.md, .py, .ipynb, comments, docstrings, etc.) must be ASCII-only. Render math through LaTeX commands (e.g. `\alpha`, `\gamma`, `\int`) rather than Unicode glyphs. Replace em/en dashes with `--`/`-`, smart quotes with straight quotes, arrows with `->`/`<-`/`<->`, and accented letters with their nearest ASCII equivalent (e.g. `Levy`, `Cizeau-Bouchaud`).
- `.agents/notes/` is literature only: each note describes a paper/result in its own terms and never references repository files. Repo derivations cite literature notes, not the reverse.
- Any script or piece of code you write (cells, throwaway scripts, validations) must time its subparts so bottlenecks are visible at a glance. Use a context-manager Timer or equivalent that prints `[label] elapsed` per block; collect timings into a running log when the run has more than a handful of stages. Treat unprofiled code as incomplete.
- Do not use the per-session memory system. All persistent material (including collaboration preferences and meta-guidance like this one) belongs in the repo, routed to one of:
  - (a) `CLAUDE.md` -- general principles, conventions, workflows.
  - (b) Markdown derivations -- specific mathematical/theoretical results.
  - (c) Script files -- validations of derivations alongside the code.
  - (d) Jupyter notebooks -- visualisations of results.
  - (e) `.agents/notes/` -- summaries of literature results essential to the project.
  - (f) `.agents/scripts/` -- standard reusable scripts intended to be invoked many times without modification.
  - (g) `.agents/temp/` -- temporary files (throwaway scripts, scratch work).
  - `.agents/` lives at the parent repo root (`extended-criticality-dnn/.agents/`), not inside `random_dnn/`.

## Project Overview

Research implementation for the paper *"Dynamical and computational properties of heavy-tailed deep neural networks"* (Qu, Wardak, Gong). The central object is a randomly-initialised MLP whose weight matrices are drawn from Levy alpha-stable distributions, parameterised by stability index `alpha` (\alpha \in [1,2]) and scale `sigma_W`. The code studies phase transitions in dynamical and computational properties across the `(alpha, sigma_W)` plane.

## `RMT.py` -- core computations (PyTorch)

All functions accept `alpha` and `sigma_W` as floats. `stable_dist_sample` draws symmetric Levy alpha-stable samples on the default torch device via the Chambers-Mallows-Stuck method (Weron form) -- the same algorithm as `scipy.stats.levy_stable.rvs`, but vectorised/GPU-native and NaN-free, so no external dependency or retry loop is needed.

CLI uses keyword args with explicit type conversion:
```bash
python RMT.py func_name arg_name arg_val arg_type [arg_name arg_val arg_type ...]
# e.g.:
python RMT.py q_star_MC alpha 1.5 float sigma_W 1.0 float
```

## `numba_cavity.py` -- localization-edge cavity (numba, complex128)

Production path for the Jacobian singular-value cavity density and the
localization edge `s_c` (where the cavity density, which counts only
delocalized states, drops below the empirical density). Reproduces the exact
Gauss-Seidel semantics of `RMT.cavity_svd_resolvent` but JIT-compiled, serial
per seed, and complex128 -- the torch cfloat cavity under-localizes (float32
rounds marginal `Im G` to zero), biasing `s_c` low by ~0.3-0.4. Parallelism is
over seeds at the process level (`numba_sv_density_multiseed`); per-seed cost
is ~67 s at `num_doublings=9` on one core, x8 per extra doubling. Densities
for edge work live on the universal log grid (`log_grid()`: 64 geometric bin
centers over [1e-2, 2e2]). Same CLI convention as `RMT.py`:
```bash
python numba_cavity.py sv_density_log_grid alpha 1.5 float sigma_W 1.0 float
```

## Output file conventions

Results are saved under `fig/` with structured subdirectory names:
```
fig/jac_cavity_svd_log_pdf/num_doublings=8;logspace_params=...;num_chis=1/
    alpha100=150;sigmaW100=100;seed=0.txt
fig/MLP_agg/width=1000;depth=50;num_realisations=50/
    alpha100=150;sigmaW100=100;seed=0;log_svdvals_mean.txt
```

`theory_submit.py` provides the save/consolidation helpers: per-realisation `.txt` outputs under `fig/` are merged into a `.npz` beside the directory (and the `.txt` files deleted) by `consolidate_arrays()`.

## Parameter conventions

- `RMT.py` / `theory_submit.py`: use actual floats (`alpha=1.5`, `sigma_W=1.0`).
- `fig/` directory naming and older scripts use integers scaled by 100 (`alpha100=150`, `g100=100`).
- `path_names.root_data` provides the cluster-aware data root.
