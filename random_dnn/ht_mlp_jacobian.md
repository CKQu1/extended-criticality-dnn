# Layerwise Jacobian singular-value distribution for heavy-tailed MLPs

This note closes the loop named at the end of `heavy_tailed_mlp.md`: it links
the heavy-tailed mean-field theory of `heavy_tailed_mlp.md` to the structured
Wishart-Levy random-matrix theory of `RMT/structured_wishart_levy.md`, and
delivers the singular-value (SV) density of the layerwise Jacobian
$J^l = D^l W^l$ at the forward fixed point $q^*$ as a single-scalar fixed
point in $Y_r(z)$.

Section 1 sets up the Jacobian and identifies the structured-matrix template.
Section 2 is the one substantive new step: the random column profile
$c(j) = |\phi'(h^l_j)|$ may be replaced by the deterministic quantile of its
limiting law (Lemma 1). Section 3 writes the resulting scalar closure
(Theorem 2 of `RMT/structured_wishart_levy.md` at $\gamma = 1$) and its
density / tail consequences. Section 4 checks the $\alpha = 2$ reduction.
Section 5 sets out the three-way validation pathway implemented in
`ht_mlp_jacobian.py` / `ht_mlp_jacobian.ipynb`. Section 6 lists scope and
open items.

## 1. Setup

A depth-$L$ MLP with constant width $N$, symmetric $\alpha$-stable weights
$W^l_{ij} = \sigma_w N^{-1/\alpha} \tilde W^l_{ij}$, $\tilde W \sim p_\alpha$,
bounded $\phi$ with $\phi(0) = 0$ (the heavy-tailed MFT scope of
`heavy_tailed_mlp.md` sec. 1, 4(c)). At a hidden layer the **layerwise
Jacobian of the postactivations** is

$$
J^l \;=\; D^l\, W^l, \qquad
D^l \;=\; \mathrm{diag}\!\big(\phi'(h^l_j)\big)_{j=1}^N, \qquad
J^l \in \mathbb{R}^{N \times N}. \qquad (1)
$$

Singular values of $J^l$ equal those of $(J^l)^{\mathsf t} = (W^l)^{\mathsf t} D^l$,
which is a heavy-tailed random matrix with **column scaling**
$(W^{\mathsf t} D)_{ij} = \tilde W^l_{ji}\, \sigma_w N^{-1/\alpha}\,
\phi'(h^l_j)$, i.e. column $j$ scaled by $\phi'(h^l_j)$. This is the
one-sided template of `RMT/structured_wishart_levy.md` Theorem 2, with
random column profile

$$
c(j) \;=\; |\phi'(h^l_j)|, \qquad j = 1, \dots, N, \qquad (2)
$$

global entry scale $\sigma_w$, and aspect ratio $\gamma = N/N = 1$ (so the
zero-atom of mass $1 - \gamma$ vanishes).

The forward MFT of `heavy_tailed_mlp.md` gives the layer-$l$ marginal of
$h^l_j$ in the wide-$N$ limit: symmetric $\alpha$-stable with scale
$S^l = (q^l)^{1/\alpha}$, where $q^l$ satisfies the heavy-tailed length map
$q^l = V_\alpha(q^{l-1}) + \sigma_b^\alpha$ (eqs. 5, 7 there). At a depth
where $q^l = q^*$ (the fixed point of $V_\alpha + \sigma_b^\alpha$), each
$h^l_j$ has marginal $h \stackrel{d}{=} S^* Z$ with $Z \sim p_\alpha$,
$S^* = (q^*)^{1/\alpha}$.

## 2. Random-to-deterministic column profile (Lemma 1)

The column profile (2) is random (it depends on the realisation of
$h^l_1, \dots, h^l_N$). The structured Wishart-Levy theorem is for a
**deterministic** profile (`RMT/structured_wishart_levy.md`, Preliminaries
and the "Deterministic $\tau$ only" remark). The bridge:

**Lemma 1 (random column profile -> deterministic quantile profile).** *Let
$X \in \mathbb{R}^{N \times M}$ have i.i.d. symmetric $\alpha$-stable entries
$\tilde W_{ij}$ and column scaling $X_{ij} = c_j \tilde W_{ij}$ with
$c_1, \dots, c_M$ i.i.d. from a distribution $F$ on $[0, \infty)$,
independent of $\tilde W$. Then in the wide-$N$ limit ($M / N \to \gamma$)
the limiting squared-SV law of $a_{N+M}^{-2} X X^{\mathsf t}$ equals the
structured Wishart-Levy law of Theorem 2 of
`RMT/structured_wishart_levy.md` with deterministic column profile equal to
the quantile function $c_*(v) = F^{-1}(v)$, $v \in [0, 1]$.*

*Proof sketch.* Three steps:

(i) **Permutation invariance.** The squared singular values of $X$ depend
on the columns of $X$ only as an unordered multiset (column permutations
realise unitary right-multiplications, which preserve singular values).
Equivalently, only the **empirical distribution**
$\hat F_M := \frac{1}{M} \sum_{j=1}^M \delta_{c_j}$ of the column scales
enters the spectrum.

(ii) **Self-averaging of the column-scale empirical distribution.** With
$c_j \stackrel{\text{iid}}{\sim} F$ and bounded $|\phi'|$ ensuring $F$ is
compactly supported (hence Glivenko-Cantelli applies), $\hat F_M \to F$
weakly almost surely as $M \to \infty$.

(iii) **Continuity of the limit spectrum in the profile.** Theorem 1 of
`RMT/structured_wishart_levy.md` is the wide-$N$ limit at a fixed
profile $\tau$; its derivation (via Theorem `weakening` of the BDG
preprint) shows that the limit spectrum depends on $\tau$ only through
$\int_0^1 |\tau(\cdot, v)|^\alpha \cdot (\text{test function})\,dv$,
which by (ii) converges to $\int_0^1 c_*(v)^\alpha \cdot (\text{test function})
\, dv$ with $c_*(v) = F^{-1}(v)$ the quantile -- that is, putting the
i.i.d. $c_j$ on a uniform grid in $[0, 1]$ ordered by quantile
gives the same limit as putting them in their drawn order.
$\square$

**Application to the Jacobian.** Apply Lemma 1 with $F$ the law of
$|\phi'(S^* Z)|$, $Z \sim p_\alpha$ (the layer-$l$ marginal at the MFT fixed
point). The deterministic column profile is

$$
c(v) \;=\; F^{-1}_{|\phi'(S^* Z)|}(v), \qquad v \in [0, 1]. \qquad (3)
$$

For $\phi = \tanh$, $|\phi'(h)| = 1 - \tanh^2(h) \in (0, 1]$, and the
quantile of $1 - \tanh^2(S^* Z)$ is bounded; numerically the quantile is
built from sorted Monte-Carlo samples of $1 - \tanh^2(S^* z_k)$,
$z_k \sim p_\alpha$. For general bounded $\phi$ the same holds: $|\phi'|$
is bounded on $\mathbb{R}$, $F$ has compact support, the quantile is
well-defined.

*Why this is the right bridge.* The existing
`RMT.py:jac_cavity_svd_log_pdf` does **not** use the quantile -- it runs a
population-dynamics scheme on the cavity equations of `RMT/hermitisation.md`
with fresh random $\chi_j = \sigma_w \phi'((q^*)^{1/\alpha} z_j)$ at each
doubling step. Lemma 1 says the limiting SV law obtained that way agrees
with the deterministic-quantile Theorem 2 -- both routes target the same
limit, and the three-way comparison in sec. 5 is a numerical witness.

## 3. Scalar closure and density (Theorem 2 of structured_wishart_levy.md)

At $\gamma = 1$ Theorem 2 of `RMT/structured_wishart_levy.md` gives a
single-scalar closure for $Y_r : \mathbb{C}^+ \to \mathcal{K}_\alpha$:

$$
z^\alpha Y_r
\;=\; \tfrac{1}{2}\, C_\alpha
   \int_0^1 c(v)^\alpha\,
   g_\alpha\!\Big( \tfrac{C_\alpha\, c(v)^\alpha}{2 z^\alpha}\, g_\alpha(Y_r) \Big)\,
   dv, \qquad (4)
$$

with $C_\alpha = i^\alpha \Gamma(1 - \alpha/2) / \Gamma(\alpha/2)$ and
$g_\alpha(y) = \int_0^\infty t^{\alpha/2 - 1} e^{-t} e^{-t^{\alpha/2} y}\,dt$
(`RMT/structured_wishart_levy.md` Preliminaries). The Cauchy-Stieltjes
transform of the squared-SV law $\nu$ is

$$
G_\nu(\zeta) \;=\; \tfrac{1}{\zeta}\, h_\alpha\!\big(Y_r(\sqrt\zeta)\big),
\qquad
h_\alpha(y) \;=\; 1 - \tfrac{\alpha}{2}\, y\, g_\alpha(y), \qquad (5)
$$

and the squared-SV density is
$\rho_\nu(t) = -\tfrac{1}{\pi t}\, \Im h_\alpha(Y_r(\sqrt t))$, with no atom
at $0$ (since $1 - \gamma = 0$). The SV density (Gram convention of
`RMT/structured_wishart_levy.md` Preliminaries) is then

$$
f_{\mathrm{SV}}(s) \;=\; 2 s\, \rho_\nu(s^2)
\;=\; -\frac{2}{\pi s}\, \Im h_\alpha\!\big(Y_r(s)\big). \qquad (6)
$$

The global $\sigma_w$ enters only through the entry scale of the underlying
$\alpha$-stable variable (the BDG quantile $a_{N+M}$ absorbs the
$N^{-1/\alpha}$); operationally this is the `entry_scale=sigma_W` argument
of `structured_wishart_levy.theoretical_structured_singular_value_curve`.

**Tail.** Theorem 1(v) at $\gamma = 1$ gives

$$
s^{1+\alpha}\, f_{\mathrm{SV}}(s) \;\longrightarrow\;
\frac{\alpha}{2}\, \sigma_w^\alpha\, \mathbb{E}_{Z \sim p_\alpha}\big[
   |\phi'(S^* Z)|^\alpha \big] \qquad (s \to \infty), \qquad (7)
$$

absorbing
$\int_0^1 c(v)^\alpha\, dv = \mathbb{E}[|\phi'(S^* Z)|^\alpha]$ (the
quantile integrates to the expectation). Equivalently the prefactor is
$B = (\alpha/2)\, \sigma_w^\alpha\, M_\alpha$ with
$M_\alpha = \mathbb{E}|\phi'(S^* Z)|^\alpha$. The same quantity appears in
the naive-decoupled slope $\chi_1|_\alpha^{\mathrm{naive}}$ of
`heavy_tailed_mlp.md` sec. 6 ($\chi_1|_\alpha^{\mathrm{naive}} = \sigma_w^\alpha
M_\alpha$); the structured-spectrum tail constant is $\alpha/2$ times this.

## 4. Reduction at $\alpha = 2$

At $\alpha = 2$ the Belinschi $a_N$ scale degenerates (`tail_amplitude` in
`belinschi_quantile_scale` vanishes because $\sin(\pi \alpha / 2)|_{\alpha=2} = 0$),
and the structured Wishart-Levy machinery does not apply via the same
$\alpha < 2$ code path. The Gaussian limit is read off directly: at
$\alpha = 2$, $W^l$ has i.i.d. $\mathcal{N}(0, \sigma_w^2 / N)$ entries, and
$J^l = D^l W^l$ is a column-scaled Gaussian matrix with diagonal
$D^l_{jj} = \phi'(h^l_j)$, $h^l_j \sim \mathcal{N}(0, q^*)$. Its squared
singular values follow a (column-scaled) Marchenko-Pastur law with
parameter $\sigma_w^2\, \mathbb{E}[\phi'(h)^2]$ at the fixed point -- which
is exactly Poole's $\chi_1$, and the Pennington-Schoenholz-Bahri (2018)
single-layer Jacobian limit. Operationally the validation pathway
(`ht_mlp_jacobian.py`) restricts $\alpha \in [1.1, 1.99]$ on the
heavy-tailed branch and treats $\alpha = 2$ via the Gaussian
(Marchenko-Pastur) branch separately.

## 5. Validation pathway

Three pathways target the same limiting SV density of $J^l$:

(P1) **Theorem-2 deterministic-quantile theory.** Build $c(v)$ from (3)
(Monte-Carlo samples of $|\phi'(S^* Z)|$ with $Z \sim p_\alpha$, sort
and treat as the empirical quantile), then call
`structured_wishart_levy.theoretical_structured_singular_value_curve(
alpha, gamma=1, tau=lambda x, y: c(y), entry_scale=sigma_W,
normalization='stable')`. Return density on a SV grid.

(P2) **Population-dynamics cavity-equation route**
(`RMT.py:jac_cavity_svd_log_pdf`). Random $\chi_j$ at each doubling step;
returns log-density on a user-supplied SV grid. Compare in **linear**
space after exponentiation (log-amplifies near-zero noise).

(P3) **Empirical SVDs** of the Jacobian, in two tiers:

  (P3a) *Synthetic.* Sample $h_j \sim S^*\cdot p_\alpha$ directly,
  build $J = D W$ with $W$ via `stable_dist_sample`, take SVDs across
  many matrix realisations. Tests Theorem 2 given $S^*$, isolated
  from MLP forward convergence.

  (P3b) *MLP-derived.* Run `RMT.py:MLP(..., fast=False)` and read
  `postjac_log_svdvals` at a depth where $q^l$ has converged to $q^*$
  (check $|q^L - q^*| / q^* < 0.05$ as a self-test). Tests the full
  pipeline.

Pass criterion: (P1, P2, P3a) overlap to bin noise on the SV histogram
across the support, and the empirical tail constant $B$ (eq. 7) matches.
(P3b) is expected to agree once the forward iterate has reached $q^*$; if
it does not, the failure is forward convergence (`heavy_tailed_mlp.md`
sec. 4(d), exponentially-small $q^*$ at small $\sigma_w$), not the
Jacobian theory.

The convention check that ties this together: `entry_scale = sigma_W` in
(P1) and `sigma_W * (2N)^{-1/\alpha}` in the SciPy-stable sampling of
(P3a) realise the same Belinschi normalisation; an empirical
match on one $(\alpha, \sigma_w)$ point is the convention witness
(implemented as a smoke test in `ht_mlp_jacobian.py`).

**Solver choice (independent-rule $h_\alpha$ for the density readout).**
(P1) is implemented through
`RMT/one_sided_wishart_levy.py:theoretical_one_sided_singular_value_curve`,
which solves the scalar field equation (4) with the shared-rule
$g_\alpha$ engine of `structured_wishart_levy.py` (so the field $Y_r$
matches the structured solver to machine precision) but reads off the
density via *independent-rule* $h_\alpha$ (its own Gauss-Laguerre
integral, as in `wishart_levy.py`). The Theorem 2 collapse does not
require the discrete sum-rule identity that forces the structured
Theorem 1 solver to use the shared-rule $h_\alpha$; the
shared-rule form has $\sim 5 \times 10^{-3}$ absolute error in
$\mathrm{Im}\,h_\alpha$ that, via the density readout
$\rho \propto \mathrm{Im}\,h / s$, was amplified at small $s$ in
earlier versions of this pipeline (using the structured curve
directly). With the independent-rule readout the small-$s$ accuracy
matches (P2) / wishart_levy across the full SV support; see
`RMT/one_sided_wishart_levy.md`.

## 6. Scope and open items

- **Layerwise, not product.** This note delivers the SV distribution of
  the per-layer Jacobian $J^l$. The product $\prod_{l=1}^L J^l$
  (governing end-to-end input perturbation and gradient propagation)
  is the heavy-tailed analogue of Pennington-Schoenholz-Bahri 2018;
  it requires a free-multiplicative composition rule for heavy-tailed
  S-transforms, deferred to a separate derivation.

- **Bounded $\phi$ only.** As in `heavy_tailed_mlp.md` sec. 4(c) the
  forward $q^*$ is infinite for unbounded $\phi$ (ReLU, linear);
  consequently the column-profile law $F$ is undefined and the Jacobian
  theory does not apply without a reparameterisation. ReLU-type
  Jacobians need either a truncation cutoff or a profile-of-the-spectral-measure
  reformulation.

- **Off-fixed-point depth.** At small $\sigma_w$ where $q^*$ is
  exponentially small (and operationally indistinguishable from $0$ on a
  finite $z$-grid; `heavy_tailed_mlp.md` sec. 4(d)), the column profile
  drifts with depth and the Jacobian theory must be evaluated at
  $S^l = (q^l)^{1/\alpha}$ rather than $S^*$. This is a per-layer change
  in the profile $c(v)$ but does not affect the closure.

- **Non-square layers.** For $\gamma = N_l / N_{l-1} \ne 1$ the atom
  $1 - \gamma$ at zero is non-trivial; Theorem 2 at general $\gamma$
  applies verbatim, just with the atom carried through to the squared-SV
  law and the SV-density tail prefactor of eq. 7 picks up the
  $\gamma / (1 + \gamma)$ factor of Theorem 1(v).

- **$\alpha = 2$.** Run the Gaussian branch (Marchenko-Pastur with
  $\chi_1$); do not attempt the heavy-tailed code path numerically at
  $\alpha = 2$ (where `belinschi_quantile_scale` vanishes).
