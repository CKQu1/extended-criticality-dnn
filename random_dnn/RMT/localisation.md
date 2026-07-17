# Singular-vector localisation for structured heavy-tailed matrices

Localisation of the singular vectors of a structured heavy-tailed random
matrix $A = (a_{ij} x_{ij})$, with $x_{ij}$ i.i.d. symmetric Levy
$\alpha$-stable and $a_{ij} \ge 0$ a deterministic entry profile, in the
operational regime $\alpha \in (1, 2)$.  The criterion is the stability of
the $\mathrm{Im}\,G = 0$ fixed point of the bipartite cavity recursion,
measured directly by population dynamics: the growth rate of the mean log
imaginary part of the resolvent.  Validation is against direct
diagonalisation of the Hermitised matrix.

Companion files:
- `localisation.py` -- the growth-rate estimator (this note's only solver).
- `structured_wishart_levy.md` -- the real-part / spectral-density closure.
- `hermitisation.md` -- bipartite Hermitisation of the singular-value problem.
- literature notes: `.agents/notes/belinschi-2008.md`,
  `.agents/notes/bordenave-2012.md`.

---

## 1. Why the density is analytic and localisation is not

The two questions one asks of the resolvent $G_{ii}(z) = [(z-H)^{-1}]_{ii}$
at $z = E + i\eta$ are computed from genuinely different functionals of its
limiting law.

- **Spectral density** is
  $\rho(E) = -\frac{1}{\pi}\langle \mathrm{Im}\,G\rangle$: a *linear*
  (first-moment) functional of the resolvent law.  The
  Belinschi-Dembo-Guionnet limit hands over the full limiting law of $G$,
  so $\langle\mathrm{Im}\,G\rangle$ is a single integral -- closed form
  (`structured_wishart_levy.md`, Thm 1(iv)).

- **Localisation** is not a moment of that law at all.  It is the
  **stability of the $\mathrm{Im}\,G = 0$ fixed point** under the cavity
  recursion (Section 3): seed an infinitesimal imaginary part at
  $\eta \to 0^+$ and ask whether the recursion grows it (delocalised --
  the spectrum at $E$ carries extended states) or contracts it (localised).
  This is a Lyapunov-exponent question about a disordered multiplicative
  process, and it has no single-integral closed form.  We therefore
  measure it directly.

Asymptotically the transition is direct (delocalised/localised): the
multifractal dimensions satisfy $D_q \in \{0, 1\}$ in the $N \to \infty$
limit, so a non-trivial measured $D_q$ is a finite-$N$ crossover
phenomenon.  Locating the edge is the complete asymptotic localisation
answer.

## 2. Setting: structured ensemble and bipartite cavity

Let $A \in \mathbb{R}^{N \times M}$, $A_{ij} = a_{ij}\, x_{ij}$, with
$x_{ij}$ i.i.d. symmetric Levy $\alpha$-stable scaled so the bulk singular
values are $O(1)$, and $a_{ij} \ge 0$ deterministic.  The singular-value
problem is the eigenvalue problem of the bipartite Hermitisation
$H = \begin{pmatrix} 0 & A \\ A^\top & 0 \end{pmatrix}$
(`hermitisation.md`): eigenvalues $\pm s_k$, and at $z = E + i\eta$ the
bipartite energy $E$ coincides with the singular value $s$ for $E > 0$.
Left/right singular vectors are the row-block / column-block restrictions
of the eigenvectors.

On the locally tree-like bipartite graph the diagonal cavity resolvents
split into a row side $G^R_i$ and a column side $G^C_j$:
$$
G^R_i = \Big(z - \sum_{j} a_{ij}^2\, x_{ij}^2\, G^C_j\Big)^{-1},
\qquad
G^C_j = \Big(z - \sum_{i} a_{ij}^2\, x_{ij}^2\, G^R_i\Big)^{-1}. \tag{1}
$$
For the one-sided row profile relevant to the MLP Jacobian, $a_{ij} = a_i$
(row nodes carry the profile, column nodes do not).

## 3. The criterion: growth rate of the mean log imaginary part

At $\eta \to 0^+$ the real part of (1) decouples and reaches its own fixed
point (the density-side closure).  Linearising in the imaginary part about
that fixed point, with $\mathrm{Im}\,G_j = \Delta_j\,|\mathrm{Re}\,G_j|^2$,
$$
\Delta^R_i = \sum_j \frac{a_{ij}^2\, x_{ij}^2}{(E - S^C_j)^2}\,\Delta^C_j,
\qquad
\Delta^C_j = \sum_i \frac{a_{ij}^2\, x_{ij}^2}{(E - S^R_i)^2}\,\Delta^R_i,
\tag{2}
$$
a linear recursion with quenched heavy-tailed bond weights.  Under
iteration $\Delta$ grows or decays exponentially; the discriminator is the
quenched Lyapunov exponent
$$
\varphi(E) \;=\; \lim_{t \to \infty} \frac{1}{t}\,
\big\langle \ln \mathrm{Im}\, G^{(t)} \big\rangle,
\qquad
\begin{cases}
\varphi(E) > 0 & \text{delocalised at } E,\\[2pt]
\varphi(E) < 0 & \text{localised at } E,
\end{cases}
\tag{3}
$$
with $t$ the sweep (tree-generation) index.  The mobility edge $s^*$ is
the zero of $\varphi$.  Because (3) averages $\ln \mathrm{Im}\,G$ -- not a
moment of it -- it is the free energy of the underlying directed
polymer, and it is exactly what population dynamics measures: propagate a
tangent imaginary channel through the real-part cavity pool with the same
disorder draws, renormalise each sweep by the geometric mean $g$, and
average $\ln g$ over sweeps (`localisation.py:_sweeps`).

**Finite-pool bias and extrapolation.**  A pool of size $P$ realises the
linear recursion as a travelling front with a population cutoff; the
measured rate obeys the Brunet-Derrida form
$$
\varphi_P = \varphi_\infty - \frac{c}{\ln^2 P},
\tag{4}
$$
so any fixed-$P$ zero is a lower bound on $s^*$.  Each $s$-point runs a
doubling ladder over pool sizes (duplicating the pool as a warm start,
short re-burn per rung) and quotes both the per-rung $\varphi_P$ (a
precision product) and the extrapolated $\varphi_\infty$ against (4) (an
accuracy product) -- `localisation.py:growth_rate_ladder`,
`bd_extrapolate`.

**Units.**  Neighbour amplitudes use the Chambers-Mallows-Stuck physical
convention $((2K)^{-1/\alpha} x)^2$ with $x$ exact symmetric
$\alpha$-stable draws, so $s$ is in physical singular-value units at every
$\alpha$ with no conversion factor.

## 4. Delocalisation baseline and the profile mechanism

Set $a_{ij} \equiv 1$.  The bipartite cavity recursion is then identical
to the symmetric heavy-tailed recursion of Bordenave-Guionnet 2012
(`.agents/notes/bordenave-2012.md`) at entry index $\mu = \alpha$ (the
squared-entry index $\alpha/2$ is internal to the recursion; there is no
Hermitisation halving).  Consequence, proven there: for $1 < \alpha < 2$
the **unstructured ensemble is delocalised throughout** -- no mobility
edge in the operational range.

Hence any genuine localisation for $\alpha \in (1, 2)$ must be
**profile-induced**.  A bounded profile does not change the entry tail
index, so it cannot move the baseline by reweighting alone; it localises
through effective *sparsification* -- a finite fraction of near-zero rows
cutting the connectivity graph.  For the MLP Jacobian this is saturation:
rows with $|\phi'(h_i)| \approx 0$ (see `ht_mlp_jacobian.md` sec. 6).
The growth-rate criterion needs no modification for this mechanism: the
profile enters (2) as the quenched row weights, and $\varphi$ responds to
sparsification directly.

## 5. Validation: direct diagonalisation

The estimator is validated against exact singular vectors of sampled
matrices (the Hermitised Jacobian at matched $(\alpha, \sigma_W)$): IPR /
$D_q$ of SVD vectors across an $N$-ladder, delocalised baseline at small
$\sigma_W$ reproducing Bordenave-Guionnet, and the measured onset compared
with the $\varphi$ zero (per-rung bound and BD-extrapolated value).
Empirical drivers: `ht_mlp_jacobian.py:mlp_jacobian_Dq_spectrum`,
`synthetic_jacobian_Dq_spectrum`, `verify_Dq_formula`.

## 6. Downstream: the heavy-tailed MLP Jacobian

The layerwise Jacobian $J^l = D^l W^l$,
$D^l = \mathrm{diag}(\phi'(h^l_i))$, is the row-profile specialisation
$a_i = \sigma_w |\phi'(h^l_i)|$ of this ensemble, with the profile law
taken at the forward fixed point $q^*$ of `heavy_tailed_mlp.md`.  The
per-cell estimator is `localisation.py:cell`; the phase-diagram sweep
driver is `.agents/scripts/phi_star_sweep.py`; results are collected in
`ht_mlp_jacobian.md` sec. 6 and `deloc_edge_of_chaos.md`.
