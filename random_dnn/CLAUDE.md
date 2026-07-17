# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working agreement

- Read-only `git` commands (`status`, `log`, `diff`, `show`, `blame`, `ls-files`, `rev-parse`, and the like) are fine without asking. Do not run any state-changing `git` command unless the user has explicitly approved it: immediately before running one, or a clearly defined batch, ask for confirmation for that specific command or batch. The user does their own `git` staging and commits; make file edits/deletions as requested (including `git rm` when retiring a file), but do not run mutating `git` commands speculatively or as routine cleanup.
- When drafting commit messages for the user, do not include a `Co-authored-by` trailer unless the user explicitly asks for it.
- Source files (.md, .py, .ipynb, comments, docstrings, etc.) must be ASCII-only. Render math through LaTeX commands (e.g. `\alpha`, `\gamma`, `\int`) rather than Unicode glyphs. Replace em/en dashes with `--`/`-`, smart quotes with straight quotes, arrows with `->`/`<-`/`<->`, and accented letters with their nearest ASCII equivalent (e.g. `Levy`, `Cizeau-Bouchaud`).
- Math rendering in responses is surface-dependent. The Claude Science web UI typesets inline (`$...$`) and display (`$$...$$`) LaTeX, so state results with LaTeX math there. The Claude Code terminal does not render LaTeX -- it shows raw `$...$` -- so in Claude Code, write math in plain text (the ASCII conventions above), not `$`-delimited LaTeX. The ASCII-only rule for source files holds on both surfaces regardless.
- `.agents/notes/` is literature only: each note describes a paper/result in its own terms and never references repository files. Repo derivations cite literature notes, not the reverse. A note may carry a self-contained companion script/notebook (e.g. `burda-2007.py` + `burda-2007.ipynb`) implementing the paper's equations in the paper's own terms; companions depend only on standard libraries and are never imported by repository code -- a repo-side validation that needs such machinery reimplements the piece it needs. Notes in `.agents/notes/inactive/` are historical record only -- new derivations must cite only notes in `.agents/notes/` proper; when a note's machinery is retired from the repo, move the note (and its preprint) into `inactive/` with a one-line status header.
- No operational log files: results live in the markdown derivations and results documents (e.g. `deloc_edge_of_chaos.md`), and history lives in git. When a result changes, update the document in place; note a superseded value with a one-line pointer when it would otherwise be re-derived (retired 2026-07: the per-campaign `*_log.md` files).
- This file may name specific files as illustrative examples; each rule must stand unambiguously without its example. Statements of current fact (locations, production paths, module names) are maintenance obligations: update them in the same change that moves, renames, or retires what they describe.
- Any script or piece of code you write (cells, throwaway scripts, validations) must time its subparts so bottlenecks are visible at a glance. Use a context-manager Timer or equivalent that prints `[label] elapsed` per block; collect timings into a running log when the run has more than a handful of stages. Treat unprofiled code as incomplete.
- Do not use the per-session memory system. All persistent material (including collaboration preferences and meta-guidance like this one) belongs in the repo, routed to one of:
  - (a) `CLAUDE.md` -- general principles, conventions, workflows.
  - (b) Markdown derivations -- specific mathematical/theoretical results.
  - (c) Script files -- validations of derivations alongside the code.
  - (d) Jupyter notebooks -- visualisations of results.
  - (e) `.agents/notes/` -- summaries of literature results essential to the project.
  - (f) `.agents/scripts/` -- standard reusable scripts intended to be invoked many times without modification.
  - (g) `.agents/temp/` -- temporary files (throwaway scripts, scratch work).
  - `.agents/` lives at `random_dnn/.agents/`.

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
density-deviation onset `s_c` (where the cavity density, which counts only
delocalized states, drops below the empirical density) -- a finite-pool
onset diagnostic; the production mobility edge is the growth-rate sweep
(see Parallel / distributed sweeps). Reproduces the exact
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

## Parallel / distributed sweeps

Phase sweeps are embarrassingly parallel over `(alpha, sigma_W)` cells;
per-cell cost is dominated by scalar scipy quadrature (`levy_stable.pdf`
inside `quad`) or the MC cavity iteration, neither of which vectorises or
uses BLAS -- speedups come from caching and processes, not threads.

Production paths (2026-07): the localisation transition is the cavity
growth-rate sweep `.agents/scripts/phi_star_sweep.py` driving
`RMT/localisation.py:cell` (growth rate of the mean log Im of the
resolvent under the linearised cavity recursion; doubling ladder +
Brunet-Derrida extrapolation; per-cell shard files on every path, so
shards are interrupt-safe; host-independent by construction, with a
deterministic per-cell rng); the truncated/conditional gain is recut at
the MC edges by `.agents/scripts/chi1_mc_recut.py` (which also rebuilds
the analytic density per cell); the empirical multi-q D_q-cut diagram is
`.agents/scripts/cstar_grid.py`.  Validation of the transition is direct
diagonalisation (`.agents/scripts/dq_ladder_jobs.py` /
`submit_dq_ladder.py`).  Results narrative: `deloc_edge_of_chaos.md`.
(Retired 2026-07: the Tarquini transfer-operator/kernel machinery, the
eta-scaling exponent, the alpha < 1 closed-form mobility edge, and their
sweeps and artifacts -- see git history.)

Shared sweep infrastructure:

- `.agents/scripts/pbs_phase_sweep.py` is the generic sharding toolkit
  (no CLI of its own): capacity-probed plans over ssh-reachable
  physics-network hosts (shared NFS home + silo: same repo, venv, and
  output paths everywhere) plus the CPU PBS queues on the physics
  headnode (defaultQ -> physics cpu-share vnodes, taiji; GPU queues
  deliberately unused).  Sweep drivers import it and build their own job
  scripts/specs (see phi_star_sweep.py).  Etiquette: check `loadavg`
  before claiming a shared host; the dynamic probe does this
  automatically (nproc minus current load minus one headroom core per
  workstation).
- The phi_star grid is alpha = 1.0-2.0, sigma_W = 0.1-3.0, both step 0.1
  (330 cells), set in `phi_star_sweep.py`.  The heavy-tailed density
  branch is alpha in (1, 2) only, so alpha = 1.00 / 2.00 columns NaN out
  on density-side quantities by construction.

## Output file conventions

Results are saved under `fig/` with structured subdirectory names:
```
fig/jac_cavity_svd_log_pdf/num_doublings=8;logspace_params=...;num_chis=1/
    alpha100=150;sigmaW100=100;seed=0.txt
fig/MLP_agg/width=1000;depth=50;num_realisations=50/
    alpha100=150;sigmaW100=100;seed=0;log_svdvals_mean.txt
```

`theory_submit.py` provides the save/consolidation helpers: per-realisation `.txt` outputs under `fig/` are merged into a `.npz` beside the directory (and the `.txt` files deleted) by `consolidate_arrays()`.

`fig/` names carry fixed computation parameters only. Swept grids are data,
not name material: a consolidated/phase-diagram npz stores its own axes
(`alpha_vals`/`sigma_W_vals`), and its name lists just the pipeline settings
(e.g. `kernel_chi1_phase_diagram;Es_norm=0.5-6.5;profile_order=32;seed=0.npz`).
One logical artifact per parameter set -- grid reruns and refinements update
it in place; exploratory or non-standard grids belong in `.agents/temp`, not
`fig/`. (Pre-2026-07 phase-diagram npz names embed grid endpoints; they are
superseded, not renamed.)

## Parameter conventions

- `RMT.py` / `theory_submit.py`: use actual floats (`alpha=1.5`, `sigma_W=1.0`).
- `fig/` directory naming and older scripts use integers scaled by 100 (`alpha100=150`, `g100=100`).
- `constants.root_data` (parent repo root, `extended-criticality-dnn/constants.py`) provides the cluster-aware data root (superseded: `path_names.py`, converted to `constants.py` and removed).
