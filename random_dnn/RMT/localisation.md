# Profile-aligned singular-vector localisation for structured Wishart-Levy matrices

## Setting and scope

Let $A \in \mathbb{R}^{N \times M}$ be a structured heavy-tailed
rectangular matrix with profile $\tau$ as in `RMT/structured_wishart_levy.md`.
The bipartite Hermitisation $H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}$
has resolvent diagonals $R^{(1)}_i(z) := (H - z)^{-1}_{ii}$ for the type-(1) side
(rows of $A$, $i = 1, \ldots, N$) and $R^{(2)}_j(z) := (H - z)^{-1}_{(N+j)(N+j)}$
for the type-(2) side (columns, $j = 1, \ldots, M$). The local density of states
(LDoS) at site $i$ on the type-(1) side is

$$
\rho^{(1)}_i(E) \;:=\; -\frac{1}{\pi}\,\lim_{\eta \to 0^+}\,\mathrm{Im}\, R^{(1)}_i(E + i\eta)
\;=\; \sum_n |u_n(i)|^2\, \delta(E - \mu_n)
$$

where $\{(\mu_n, u_n)\}$ are the eigenvalues / eigenvectors of $H$
(left singular vectors of $A$ correspond to $u_n$ restricted to the
type-(1) block, after the bipartite-mass split of `RMT/hermitisation.md`).
Analogous statements hold on the type-(2) side for right singular vectors.

This note derives a profile-aligned **localisation observable** -- a
non-trivial readout of the structured-Wishart-Levy solution
$(Y_r, Y_c)$ that detects whether singular vectors concentrate on
particular regions of the row/column profile. It is distinct from the
intrinsic localisation of Bordenave-Guionnet in unstructured heavy-tailed
matrices, which is a finite-$N$ fluctuation phenomenon (cf.
`.agents/notes/bordenave-2012.md`) that the BDG-style deterministic limit
does not see -- this is discussed in Section 4.

## 1. The per-position resolvent from $(Y_r, Y_c)$

The structured-Wishart-Levy solution $(Y_r, Y_c) :
\mathbb{C}^+ \to L^\infty([0,1]; \mathcal{K}_\alpha)$ from Theorem 1 of
`structured_wishart_levy.md` encodes the per-position limit of the
resolvent diagonals. Specifically, for $x = i/N$ on the row axis and
$y = j/M$ on the column axis, the BDG-style limit relations are

$$
\boxed{\quad
G^{(1)}_x(z) \;=\; \frac{1}{z}\, h_\alpha\!\big(Y_r(x, z)\big),
\qquad
G^{(2)}_y(z) \;=\; \frac{1}{z}\, h_\alpha\!\big(Y_c(y, z)\big),
\quad} \qquad (1)
$$

where $G^{(1)}_x(z), G^{(2)}_y(z)$ are the (deterministic-in-the-limit)
type-(1) and type-(2) resolvent diagonals at row position $x$ and
column position $y$, and $h_\alpha(y) = \int_0^\infty e^{-t}\, e^{-t^{\alpha/2} y}\, dt$
is the integral kernel of Theorem 1. Integrating (1) over $x$ recovers
the spectral density via Theorem 1(iv).

Equation (1) is the load-bearing identification for what follows: the
BDG-style field $Y_r(x, z)$ contains, in transformed form, the per-row
local resolvent at *bipartite* spectral coordinate $z$, i.e. at
singular value $s = \mathrm{Re}\, z$ of $A$ (eigenvalues of $H$ are
$\pm s_k$, so the bipartite-$z$ coordinate and the singular-value
coordinate coincide for $\mathrm{Re}\, z > 0$). The corresponding per-row
LDoS in the singular-value variable $s$ is

$$
\rho^{(1)}_x(s) \;=\; -\frac{1}{\pi s}\,\mathrm{Im}\, h_\alpha\!\big(Y_r(x, s + i 0^+)\big)
\qquad (s > 0).
$$

This matches Theorem 1(iv) of `structured_wishart_levy.md` upon
$\zeta = s^2$ change of variable: the squared-singular-value density
$\rho_\nu(t) = -\frac{1}{\pi t}\, \mathrm{Im} \int_0^1 h_\alpha(Y_r(x, \sqrt t))\, dx$
and the singular-value density satisfy $f_{\text{SV}}(s) = 2s\, \rho_\nu(s^2)
= -\frac{2}{\pi s}\,\mathrm{Im}\int_0^1 h_\alpha(Y_r(x, s))\, dx$,
which integrates the per-row $\rho^{(1)}_x(s)$ (up to a factor $2$ from
the bipartite mass split, cf. `RMT/hermitisation.md` eq. (10)). The
load-bearing point is that $\rho^{(1)}_x(s)$ is well-defined per row $x$,
and its $x$-variation is the seat of profile-aligned localisation.

## 2. Profile-aligned localisation observable

For singular vectors of $A$ to localise on a particular region $R \subseteq
[0, 1]$ of the row profile, the per-row LDoS $\rho^{(1)}_x(s)$ must be
*concentrated* on $x \in R$ for singular values $s$ corresponding to those
vectors. This is precisely captured by the variation of $\rho^{(1)}_x(s)$
across $x$, which the BDG framework retains.

Define, for $q \in (0, 1)$ and $s$ in the singular-value spectrum,

$$
\boxed{\quad
M_q^{(1)}(s) \;:=\; \int_0^1 \big(\rho^{(1)}_x(s)\big)^q\, dx,
\qquad
\bar\rho^{(1)}(s) \;:=\; \int_0^1 \rho^{(1)}_x(s)\, dx
\quad} \qquad (2)
$$

with analogous $M_q^{(2)}, \bar\rho^{(2)}$ on the column side. By Jensen's
inequality (concavity of $x \mapsto x^q$ for $q < 1$),

$$
M_q^{(1)}(s) \;\le\; \big(\bar\rho^{(1)}(s)\big)^q,
$$

with equality if and only if $\rho^{(1)}_x(s)$ is constant in $x$
(uniform LDoS, no profile-aligned localisation). The **localisation index**

$$
\boxed{\quad
\ell_q^{(1)}(s) \;:=\; 1 \,-\, \frac{M_q^{(1)}(s)}{\big(\bar\rho^{(1)}(s)\big)^q}
\;\in\; [0, 1]
\quad}\qquad (3)
$$

is $0$ for uniform LDoS (profile-delocalised) and approaches $1$ for
maximally concentrated LDoS (singular vectors localised on a vanishing
fraction of rows). The natural choice $q = \alpha/2$ parallels the
Bordenave-Guionnet observable; other $q \in (0, 1)$ give related but
generally weaker diagnostics (the optimal $q$ depends on the tail
exponent of the LDoS distribution across rows).

For the column side: $\ell_q^{(2)}(s)$ defined analogously diagnoses
right-singular-vector localisation on column-profile regions.

## 3. Profile-conditional refinement

In the structured setting the row position $x$ is more informative than a
single number per spectral value: the profile $\tau$ partitions $[0, 1]$
into regions of constant or slowly varying $|\tau(x, \cdot)|^\alpha$, and
localisation is naturally diagnosed *within* each region. Concretely,
for a profile-stratification $[0, 1] = \bigsqcup_k S_k$ with strata $S_k$
on which $|\tau(x, \cdot)|^\alpha$ is approximately constant, define

$$
\rho^{(1)}_k(s) \;:=\; \frac{1}{|S_k|}\int_{S_k} \rho^{(1)}_x(s)\, dx,
\qquad
M_{q,k}^{(1)}(s) \;:=\; \frac{1}{|S_k|}\int_{S_k} \big(\rho^{(1)}_x(s)\big)^q dx.
$$

The within-stratum localisation index
$\ell_{q,k}^{(1)}(s) := 1 - M_{q,k}^{(1)}(s) / (\rho^{(1)}_k(s))^q$
detects localisation *within* the constant-profile region $S_k$.
Cross-stratum localisation is detected by comparing $\rho^{(1)}_k(s)$
across $k$: if certain strata have $\rho^{(1)}_k(s) \gg$ others, singular
vectors at singular value $s$ concentrate on those strata.

The two readouts -- within-stratum $\ell_{q,k}$ and cross-stratum
$\{\rho^{(1)}_k\}_k$ -- decompose the full profile-aligned localisation
into "fine-grained" (within a constant-profile patch) and "coarse-grained"
(profile-feature-aligned) components.

## 4. What this captures and what it does not

**This observable captures**: profile-aligned localisation -- singular
vectors concentrating on regions of the row/column profile where the
LDoS is non-uniform. For the MLP Jacobian (Section 6), this is
singular vectors aligning with neurons in particular regimes of $|\phi'|$
(saturating vs linear), which is the operationally important
localisation phenomenon.

**This observable does *not* capture**: intrinsic Bordenave-Guionnet
localisation in the *unstructured* case $\tau \equiv 1$. In that case
$Y_r(x, z) = Y_r(z)$ is $x$-independent, $\rho^{(1)}_x(s)$ is constant in
$x$, and $\ell_q^{(1)}(s) = 0$ identically -- yet Bordenave-Guionnet
predict and prove genuine eigenvector localisation in the heavy-tailed
tails of the spectrum for $\alpha < 2/3$ (cf.
`.agents/notes/bordenave-2012.md`). The reason: that localisation is a
*finite-$N$ fluctuation* of $\mathrm{Im}\, R^{(1)}_i(s + i\eta)$ across
sites $i$, which the deterministic BDG limit by construction smooths
out. To recover it in our framework we would have to track the
*variance* (or higher moments) of $R^{(1)}_i$ across $i$ at finite $N$,
which is a sub-leading-in-$N$ phenomenon outside the
structured-Wishart-Levy theorem.

For the structured heavy-tailed MLP Jacobian, intrinsic Bordenave-Guionnet
localisation and profile-aligned localisation can in principle coexist;
the operationally important one for trainability and gradient
interpretation is the profile-aligned one, which the observables (2)-(3)
deliver.

## 5. Numerical evaluation

The structured-Wishart-Levy solver in `structured_wishart_levy.py`
computes $Y_r(x, z), Y_c(y, z)$ on a discretisation of $[0, 1]$ at given
$z = E + i\eta$ (typically with small $\eta > 0$ regulariser). The
localisation observable evaluation adds a single reduction step:

1. Pull $Y_r(x_k, z), Y_c(y_k, z)$ at the discretisation nodes $x_k, y_k$.
2. Evaluate $h_\alpha(Y_r(x_k, z))$ and $\mathrm{Im}\, h_\alpha(Y_r(x_k, z))$
   for each node (the spectral-density code path already does this for the
   $x$-integral).
3. Take per-node LDoS $\rho^{(1)}_{x_k}(s) = -\frac{1}{\pi s}\, \mathrm{Im}\,
   h_\alpha(Y_r(x_k, s + i\eta))$.
4. Numerical integrals over $x_k$: $\bar\rho^{(1)}(s)$ (existing) and
   $M_q^{(1)}(s) = \sum_k w_k\, (\rho^{(1)}_{x_k}(s))^q$ for $q = \alpha/2$.
5. Localisation index $\ell_q^{(1)}(s) = 1 - M_q^{(1)}(s) /
   (\bar\rho^{(1)}(s))^q$.
6. Sweep $s$ across the spectrum; sweep $(\alpha, \sigma_w$ or profile parameters)
   for a localisation phase diagram.

No new equations to solve; one extra reduction step over the same nodes.

## 6. Connection to the MLP Jacobian

In the heavy-tailed MLP setting of `heavy_tailed_mlp.md`, the layer-wise
Jacobian $J^l = D^l W^l$ has $D^l = \mathrm{diag}(\phi'(h^l))$ on the
right. Its singular value spectrum is captured by the structured
Wishart-Levy law with column profile

$$
c(y) \;=\; |\phi'(h^{l-1}(y))|
$$

at the heavy-tailed length-map fixed point $q^*$ (cf. `heavy_tailed_mlp.md`
Section 4(e)). The corresponding $Y_c(y, z)$ field thus encodes how the
right-singular-vector LDoS varies with the column-position $y$, i.e.,
with the *activation regime* of the previous-layer pre-activation: $y$
in a regime where $\phi'(h^{l-1}(y)) \approx 1$ (linear) vs $\approx 0$
(saturated) gives qualitatively different LDoS.

Profile-aligned right-singular-vector localisation (the column-side
observable $\ell_q^{(2)}(s)$) at a given $E$ in the Jacobian spectrum
therefore answers: do right singular vectors of $J^l$ at this singular
value concentrate on neurons whose previous-layer pre-activations are in
the linear regime, the saturated regime, or distributed uniformly?
Heavy-tailed weights at $\alpha < 2$ are expected to align localisation
with the saturated regime (small $|\phi'|$), where the structured tail
of the matrix amplifies rare large entries; verifying this empirically
is the load-bearing follow-on.

The full input-output Jacobian $J = J^L \cdots J^1$ requires composing
the per-layer localisation via free product (analogous to S-transform
composition for the spectrum); this is a separate derivation
(`random_dnn/heavy_tailed_mlp_jacobian.md`, to be written).

## 7. Open questions

1. *Optimal $q$.* The choice $q = \alpha/2$ parallels Bordenave-Guionnet
   for the symmetric case. Whether the Jensen-gap (3) is maximised at
   $q = \alpha/2$ for our structured case, or at a different $q$
   depending on the tail of the LDoS-across-rows distribution, is open.
   For practical use a $q$-sweep at one $(\alpha, \tau)$ point should
   identify the diagnostic $q$.
2. *Equation (1) verification.* The identification
   $G^{(1)}_x(z) = z^{-1} h_\alpha(Y_r(x, z))$ is read off from
   Theorem 1(iv) of `structured_wishart_levy.md` (where the row-integral
   of the same expression gives the spectral density); the per-row form
   should be checked against the underlying BDG paper to make sure no
   row-average is implicit. A spot-check on the unstructured
   (Corollary 1) reduction case where $Y_r, Y_c$ collapse to scalars and
   $G^{(1,2)}_x$ to constants should suffice.
3. *Intrinsic vs profile-aligned coexistence.* For the MLP Jacobian, the
   structured-Wishart-Levy framework captures profile-aligned
   localisation. Intrinsic Bordenave-Guionnet localisation (in the
   fluctuations) coexists in principle; whether it is operationally
   relevant for the heavy-tailed MLP at $\alpha > 2/3$ (where intrinsic
   localisation is not proven) is open.

---

## Next step

Implementation: extend `structured_wishart_levy.py` with a post-solve
reduction computing $\ell_q^{(1)}(s), \ell_q^{(2)}(s)$ from $Y_r, Y_c$
and the column profile. Validate on the unstructured Corollary 1 case
(localisation index $\equiv 0$, by construction) and on a step-profile
$c(y) = c_1 \mathbf{1}_{[0, 1/2]} + c_2 \mathbf{1}_{(1/2, 1]}$ with
$c_1 \ne c_2$ (localisation index $> 0$ expected, concentrated on the
larger-$c$ stratum). Then sweep $(\alpha, c_2/c_1)$ for a phase diagram.
