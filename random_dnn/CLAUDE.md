# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

- The user does their own `git` staging and commits. Make file edits/deletions as requested (including `git rm` when retiring a file), but do not run `git add`/`git commit`, and do not surface reminders or warnings about staging or commit state.
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

All functions accept `alpha` and `sigma_W` as floats. Uses `torchlevy` for GPU-accelerated Levy sampling, wrapped with a NaN-retry loop in `stable_dist_sample`.

CLI uses keyword args with explicit type conversion:
```bash
python RMT.py func_name arg_name arg_val arg_type [arg_name arg_val arg_type ...]
# e.g.:
python RMT.py q_star_MC alpha 1.5 float sigma_W 1.0 float
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
