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

## 8. One-sided specialisation $|\tau(x, y)| = c(y)$

For the Theorem 2 case (one-sided profile, `RMT/one_sided_wishart_levy.md`),
the row field collapses to a scalar: $Y_r(x, z) = Y_r(z)$, $x$-independent.
By (1), $G^{(1)}_x(z) = z^{-1} h_\alpha(Y_r(z))$ is $x$-independent as well,
so the per-row LDoS is constant in $x$:

$$
\rho^{(1)}_x(s) \;=\; -\frac{1}{\pi s}\, \mathrm{Im}\, h_\alpha(Y_r(s + i 0^+))
\qquad \text{(independent of } x \text{)}.
$$

Consequently
$$
\boxed{\quad \ell_q^{(1)}(s) \;\equiv\; 0 \quad \text{for all } s, q
\qquad \text{(one-sided case).} \quad}
$$
Left singular vectors of a one-sided structured Wishart-Levy matrix are
**profile-uniform**: there is no row-axis structure for them to localise
against, and the BDG deterministic limit reflects this exactly. (Intrinsic
Bordenave-Guionnet localisation may still exist for left vectors as a
finite-$N$ effect; the structured-deterministic framework does not see
it, per Section 4.)

All profile-aligned localisation in the one-sided case lives in the
**column-side observable $\ell_q^{(2)}(s)$**, via the slaved column field

$$
Y_c(y, z) \;=\; \frac{C_\alpha\, c(y)^\alpha}{(1+\gamma)\, z^\alpha}\, g_\alpha(Y_r(z)),
\qquad (8)
$$

(Theorem 2(i) of `RMT/one_sided_wishart_levy.md`). The per-column LDoS is

$$
\rho^{(2)}_y(s) \;=\; -\frac{1}{\pi s}\, \mathrm{Im}\, h_\alpha\!\big(Y_c(y, s + i 0^+)\big),
$$

with $Y_c(y, z)$ explicit in $c(y)$ via (8), and the column localisation
index follows from (3):

$$
\boxed{\quad
\ell_q^{(2)}(s)
\;=\; 1 \,-\, \frac{\displaystyle\int_0^1 \big(\rho^{(2)}_y(s)\big)^q\, dy}
{\Big(\displaystyle\int_0^1 \rho^{(2)}_y(s)\, dy\Big)^q}. \quad} \qquad (9)
$$

Two structurally clean features:

(a) *No extra solve.* Equation (9) requires only the already-computed
    scalar $Y_r(z)$ from `one_sided_wishart_levy.py`'s Theorem 2 scalar
    closure; $Y_c(y, z)$ is reconstructed pointwise from (8) at the
    Gauss-Legendre $y$-nodes used for the spectral-density quadrature.
    One extra Laguerre evaluation of $h_\alpha$ per $y$-node per $s$ on
    the SV grid.

(b) *Sees the profile signature directly.* Through (8),
    $Y_c(y) \propto c(y)^\alpha$, so $\rho^{(2)}_y \propto |\Im h_\alpha
    (c(y)^\alpha \cdot \mathrm{anchor})|$ varies with $y$ in a way
    inherited from the column profile. Jensen-gap is non-zero whenever
    $c(\cdot)$ is non-constant.

**Recovery of the constant-$c$ degenerate corner.** For
$c \equiv \text{const}$ (Wigner/plain-square corner, Corollary 1 of
`RMT/structured_wishart_levy.md`), $Y_c(y, z)$ is $y$-independent,
$\rho^{(2)}_y(s)$ is constant, and $\ell_q^{(2)}(s) \equiv 0$ -- profile
delocalised, as required.

**Heavy-tailed MLP Jacobian application.** With
$c(v) = F^{-1}_{|\phi'(S^* Z)|}(v)$, $Z \sim p_\alpha$ (the Jacobian
column profile of `ht_mlp_jacobian.md`), $c(v)$ rises sharply from $\approx 0$
at $v \to 0$ (saturated $\tanh$) to $\approx 1$ at $v \to 1$ (linear regime).
The $y$-variation of $\rho^{(2)}_y(s)$ then encodes whether right singular
vectors of $J^l$ at SV $s$ concentrate on saturated neurons (small $c$) or
linear-regime neurons (large $c$). Per (9), $\ell_q^{(2)}(s) > 0$ identically
in this heavy-tailed Jacobian setting; its $s$-dependence and the sign of the
profile-side concentration (small-$c$ vs large-$c$ end of $[0, 1]$) is the
operationally interesting follow-on diagnostic, implemented as
`localisation.localisation_index_curve` (implemented in `RMT/localisation.py`).

## 9. Heavy-tail multifractal $D_q$ from the BDG field (planning skeleton)

This section is currently a planning skeleton, not a derivation. It
records the structural setup for deriving the multifractal exponent
$D_q(\lambda)$ from the BDG cavity field $Y_r(z)$ via two parallel
formulations of the inverse participation ratio. The actual analytic
derivations are deferred until each load-bearing step is settled.

$\ell_q$ of sections 2-8 above is a **profile-aligned mean-LDoS**
observable -- it captures position-axis variation of the deterministic
mean LDoS $\rho^{(i)}_x(s)$ and gives $\ell_q^{(1)} \equiv 0$ for the
one-sided case. It is *not* the heavy-tail multifractal exponent $D_q$
of the eigenvectors. $D_q$ instead captures finite-$N$ heavy-tail
fluctuations of the local resolvent across sites at fixed position,
and is non-trivial for both left and right singular vectors of one-sided
structured matrices. Empirical $D_2$ from MLP-Jacobian SVDs
(`RMT.MLP_agg`) is well-defined for both sides, and this section sets
up the theory to match.

### 9.0 Setup: the cavity-resolvent distribution from $Y_r$

From the Belinschi-Dembo-Guionnet 2009 / Ben-Arous-Guionnet 2007
construction (cf. Belinschi `wishart-arxiv-rev.tex`, Prop.
`projectionoflimitpoint`), the limiting distribution $\mu^z_r$ of
the resolvent diagonal $G^r_{ii}(z)$ on the row side is the
**pushforward** of an explicit complex $\alpha/2$-stable distribution
$P^{\mu^z_r}$ on $\mathbb{C}^-$, parameterised by the scalar
$\widehat{X}_r(z) := \Gamma(1-\alpha/2) \sum_s |\sigma_{rs}|^\alpha \Delta_s X_s(z)$
(linear combination of the field scalars $X_s(z) = \int x^{\alpha/2}\,d\mu^z_s$,
which in turn are explicit in $Y_s(z) = (-z)^{-\alpha/2} X_s(z)$), under
the rational map $\Sigma \mapsto -1/(z - \Sigma)$:

$$
\mu^z_r \;=\; \big(\Sigma \mapsto -1/(z - \Sigma)\big)_\ast\, P^{\mu^z_r},
\qquad
\widehat{P^{\mu^z_r}}(t) \;=\; \exp\!\big(-\Gamma(1-\alpha/2)(it)^{\alpha/2}\,
\widehat{X}_r(z)\big).
$$

This is the rigorous content of "$Y_r$ uniquely determines the cavity
resolvent distribution," via the one-scalar parameterisation of the
underlying complex stable law. Both formulations of $D_q$ below are
read off from this pushforward construction without additional ansatz.

### 9A. Formulation via $\mathbb{E}|G|^q$ (absolute-value moments)

**Spectral identity.** From the pole structure
$G_{ii}(\lambda_k + i\eta) \sim |v_k(i)|^2/(i\eta) + R$, the IPR satisfies
$$
\bar I_q(\lambda) \;=\; \lim_{\eta \to 0^+}\, \frac{\eta^q}{\rho(\lambda)}\,
\mathbb{E}_{G \sim \mu^{\lambda + i\eta}_r}\big[|G|^q\big] \qquad (?)
$$
(needs justification of the spectral-average vs cavity-ensemble
exchange; load-bearing step **9A.1**.)

**Cavity-ensemble moment.** With $G = -1/(z - \Sigma)$,
$$
\mathbb{E}|G|^q \;=\; \mathbb{E}_{\Sigma \sim P^{\mu^z_r}}\big[|z - \Sigma|^{-q}\big].
$$

**Load-bearing analytic step 9A.2.** Reduce
$\mathbb{E}|z - \Sigma|^{-q}$ to a closed-form integral over $Y_r$ (or
$\widehat{X}_r$). Status by $q$:

**$q = \alpha/2$ anchor (partial).** $\mathbb{E}|G|^{\alpha/2}$ is the
**absolute $\alpha/2$-th moment** of $\mu^z_r$, which Cizeau-Bouchaud
sec. 7 identify as their "effective scale" parameter
$C(z) = (1/N)\sum_i |G_{ii}|^{\alpha/2}$ -- the scale of the heavy-tail
CLT for $\Sigma_i = \sum_j H_{0j}^2 G_{jj}$. **Key clarification:**
Belinschi's $X_r(z) := \int x^{\alpha/2}\, d\mu^z_r(x)$ is the
**complex** $\alpha/2$-moment using the principal branch
$x^{\alpha/2} = |x|^{\alpha/2}\, e^{i (\alpha/2) \arg x}$, so
$$
X_r(z) \;=\; \int |x|^{\alpha/2}\, e^{i (\alpha/2)\, \arg x}\, d\mu^z_r(x)
\;\ne\; \int |x|^{\alpha/2}\, d\mu^z_r(x) \;=\; C(z).
$$
These two are different functionals: $X_r$ is what closes the BAG field
equation; $C$ is the CB effective scale and equals $\mathbb{E}|G|^{\alpha/2}$.
They differ through the angular distribution of $\arg G$ over $\mu^z_r$.
Concretely, since $\mu^z_r$ is supported on $\mathbb{C}^-$ (so $\arg G
\in [-\pi, 0]$), $X_r$ has both real and imaginary parts; the imaginary
part of $X_r$ is $\int |x|^{\alpha/2} \sin(\tfrac{\alpha}{2}\arg x)\,d\mu^z_r$,
and (with $\arg x \le 0$) $\sin(\tfrac{\alpha}{2}\arg x) \le 0$ on the
support. So $\mathrm{Im}\,X_r \le 0$, $\mathrm{Re}\,X_r$ involves
$\cos$, and $C(z) = |x|^{\alpha/2}$ alone -- distinct.

The cleanest statement we can make from the BAG/Belinschi pushforward:
$$
\mathbb{E}|G|^{\alpha/2} \;=\; \mathbb{E}_{\Sigma \sim P^{\widehat{X}_r}}
\big[|z - \Sigma|^{-\alpha/2}\big], \qquad (11)
$$
where $\Sigma$ is complex $\alpha/2$-stable on $\mathbb{C}^-$ with CF
parameter $\widehat{X}_r(z)$. Whether (11) reduces to a single-Laguerre
integral $g_{\alpha, \alpha}(Y_r) = g_\alpha(Y_r)$ (which appears in
the field equation) or requires the 2-D pushforward is the **open
analytic step**. The single-Laguerre identity Belinschi `eq:gziden`
delivers $\mathbb{E}[(z - \Sigma)^{-\alpha/2}] = (-z)^{-\alpha/2}
g_\alpha(Y_r)$ -- the *complex* moment, equal to $X_r(z)$ -- not the
absolute moment. So at $q = \alpha/2$ specifically, **9A.2 does not
collapse to a single Laguerre** by the existing identity.

**Candidate routes to evaluate (11) (and general $q$):**

- Mellin: $|w|^{-q} = \frac{1}{\Gamma(q/2)} \int_0^\infty s^{q/2-1}
  e^{-s|w|^2}\,ds$. Issue: $|w|^2 = w\bar w$ couples $w$ and its
  conjugate; the expectation $\mathbb{E}[e^{-s|z-\Sigma|^2}]$ over
  stable $\Sigma$ is *not* the stable Laplace transform but a
  Gaussian-like exponent of $\Sigma$, which does not factor against
  the stable CF.
- 2-D Fourier: $|w|^{-q} \propto \int_{\mathbb{R}^2} e^{i\langle t, w\rangle}\,
  |t|^{q-2}\,dt$ (for $0 < q < 2$ in $\mathbb{R}^2$). Combined with
  $\widehat{P^{\widehat{X}_r}}(t)$, gives a 2-D integral of
  $|t|^{q-2}\, e^{i\langle t, z\rangle}\, \exp(-\Gamma(1-\alpha/2)
  (it)^{\alpha/2}\widehat{X}_r)$ over $\mathbb{R}^2$. Polar coords
  in $t$ may collapse one direction; to be attempted.
- Direct connection to Cizeau-Bouchaud's self-consistency for $C(z)$:
  if their derivation is rigorous (the user has expressed doubt; we
  do not rely on it), it gives a self-consistency equation for $C(z)$
  directly in terms of $\rho(z)$ and stable densities, *not* in terms
  of $Y_r$ explicitly. Whether $C(z) = \mathbb{E}|G|^{\alpha/2}$ can
  be derived as a closed functional of $Y_r$ (the field) without
  going through CB's controversial steps is the open question for 9A.

**$q \ne \alpha/2$ (not derived).** Same 2-D-integral fallback applies.
The two anchors ($q = \alpha/2$ for 9A here, $q = 1$ for 9B above)
together cover the "scale" and "density" moments; general $q$ moments
require either the 2-D pushforward route or a new identity not in
Belinschi.

**$D_q$ readout.** Once $\mathbb{E}|G|^q$ is closed in $Y_r$, the IPR
follows from the spectral identity (step 9A.1), and $D_q$ is the
finite-$N$ $\log N$ slope, with critical $q$ set by the heavy tail of
$|G|$ inherited from the $\alpha/2$-stable distribution of $\Sigma$.

### 9B. Formulation via $\mathbb{E}(-\mathrm{Im}\,G)^q$ (BG fractional-moment)

**Spectral identity.** Standard:
$|v_\lambda(i)|^2 \stackrel{d}{=} -\mathrm{Im}\,G_{ii}(\lambda + i 0^+) / (N\pi\rho(\lambda))$
(Lehmann), so
$$
I_q(\lambda) \;=\; \frac{1}{(N\pi\rho)^q}\, \sum_i \big(-\mathrm{Im}\, G_{ii}\big)^q
\;\sim\; \frac{N^{1-q}}{(\pi\rho)^q}\, \mathbb{E}_{\mu^{\lambda + i 0^+}_r}
\big[(-\mathrm{Im}\,G)^q\big]
$$
in the wide-$N$ self-averaging limit (for $q$ where the cavity-ensemble
moment is finite). Load-bearing step **9B.1**: justify the limit
$\eta \to 0^+$ and the cavity-ensemble self-averaging at the same scale
that gives the spectral density (i.e., a stronger statement than the
BAG mean-density convergence).

**Cavity-ensemble moment.** With $G = -1/(z - \Sigma)$,
$$
-\mathrm{Im}\,G \;=\; \frac{\eta - \mathrm{Im}\,\Sigma}
{(E - \mathrm{Re}\,\Sigma)^2 + (\eta - \mathrm{Im}\,\Sigma)^2} \;>\; 0
$$
(since $\Sigma \in \mathbb{C}^-$). Then
$$
M_q^{(\mathrm{Im})}(z) \;:=\; \mathbb{E}_{\Sigma \sim P^{\mu^z_r}}
\bigg[\bigg(\frac{\eta - \mathrm{Im}\,\Sigma}
{(E - \mathrm{Re}\,\Sigma)^2 + (\eta - \mathrm{Im}\,\Sigma)^2}\bigg)^q\bigg].
$$

**Load-bearing analytic step 9B.2.** Closed-form for $M_q^{(\mathrm{Im})}$
in terms of $Y_r$. Status by $q$:

**$q = 1$ anchor (derived).** $\mathrm{Im}$ is linear, so
$\mathbb{E}(-\mathrm{Im}\,G^r_{ii}) = -\mathrm{Im}\,\mathbb{E}\,G^r_{ii}$.
From Belinschi `eqG` / BAG `eqGint`,
$\mathbb{E}\,G^r_{ii}(z) = z^{-1}\, h_\alpha(Y_r(z))$ (the per-row
Stieltjes contribution at row position $r$; at one-sided $|\tau|=c(y)$
with $Y_r(x, z) = Y_r(z)$ scalar, $z\mapsto\sqrt\zeta$ gives Theorem 2
of `one_sided_wishart_levy.md` eq. 5). Hence
$$
\mathbb{E}(-\mathrm{Im}\,G^r_{ii})(z = s + i 0^+)
\;=\; -\frac{1}{s}\, \mathrm{Im}\, h_\alpha(Y_r(s + i 0^+))
\;=\; \pi\, \rho^{(r)}(s), \qquad (10)
$$
identifying $\mathbb{E}(-\mathrm{Im}\,G) / \pi$ with the per-row spectral
density $\rho^{(r)}(s)$ from sec. 1. So at $q = 1$ the BG-style fractional
moment **is** $\pi$ times the spectral density, recovered from $Y_r$ by
the existing $h_\alpha$ readout. Internal consistency: the $q = 1$ case
of the formulation 9B reduces to the existing density curve, no new
quantity to compute.

**$q \ne 1$ (not derived).** $\mathrm{Im}(\cdot)^q$ is non-analytic for
$q \ne 1$, so the Belinschi `eq:gziden` family (which gives analytic
moments $\mathbb{E}[(z-\Sigma)^{-\beta/2}] = $ single Laguerre over
$Y_r$) does not directly produce $\mathbb{E}(-\mathrm{Im}\,G)^q$. The
fallback is a 2-D pushforward integral over the explicit complex
$\alpha/2$-stable density of $\Sigma$ on $\mathbb{C}^-$ (obtained as
the 2-D inverse Fourier of $\widehat{P^{\mu^z_r}}(t) =
\exp(-\Gamma(1-\alpha/2)(it)^{\alpha/2} \widehat{X}_r(z))$). Polar
decomposition $\Sigma = R e^{i\phi}$ with $R > 0$,
$\phi \in (-\pi, 0]$ may collapse one of the two dimensions if the
stable density factorises favourably; this is open and to be attempted
when 9B is taken past the anchor.

**$D_q$ readout.** Same finite-$N$ $\log N$-slope formula as in 9A,
with critical $q_c$ set by the heavy tail of $-\mathrm{Im}\,G$ inherited
from the $\alpha/2$-stable tail of $\Sigma$.

### 9C. Cross-validation: A and B must agree on $D_q$

Both 9A and 9B target the same eigenvector IPR via two different
intermediate observables. In the wide-$N$, $\eta \to 0$ limit they must
give the same $D_q$ (since $D_q$ is a property of the eigenvector
distribution, not of which functional we use to read it off). At
finite $N$ and finite $\eta$ they will differ by $O(1)$ corrections.
Cross-validation steps:

- (9C.1) *Internal consistency at $q = 1$*: $\mathbb{E}|G|$ vs
  $|\mathbb{E}\,G|$ are different in general; $\mathbb{E}(-\mathrm{Im}\,G) =
  \pi\rho$ is special. Use this to check that 9A.2 and 9B.2 are
  consistent with eigenvector normalisation $I_1 = 1$.
- (9C.2) *Tail-exponent matching*: the critical $q_c$ for both
  observables should derive from the same $\alpha/2$-stable tail of
  $\Sigma$. Verify analytically.
- (9C.3) *Constant-$c$ reduction*: at $c \equiv 1$ (Wigner-Levy
  / Wishart-Levy degenerate corner), both 9A and 9B should reduce to
  the standard (Cizeau-Bouchaud predicted / Bordenave-Guionnet
  rigorous) $D_q$ values.

### 9D. Application to the heavy-tailed MLP Jacobian

For the structured one-sided $c(y)$, the row-side $D_q^{(1)}(\lambda)$
is set by the single field $Y_r(z)$; the column-side $D_q^{(2)}(\lambda)$
involves the $y$-dependent slaved $Y_c(y, z)$, with the column profile
$c(y)^\alpha$ entering the tail amplitude of the column self-energy.
Expected predictions to compare against `RMT.MLP_agg`'s `D2_mean`:

- $D_2^{(\text{left})}(\lambda)$ and $D_2^{(\text{right})}(\lambda)$
  as a function of singular value $\lambda$ at fixed $(\alpha, \sigma_w)$.
- Asymmetry $D_q^{(\text{right})} - D_q^{(\text{left})}$ in the Jacobian
  case as a structural signature of the heavy-tailed column profile
  $c(v) = F^{-1}_{|\phi'(S^* Z)|}(v)$.

### 9E. Status (what is and isn't established)

- **Established (from Belinschi 2009 / BAG 2007)**: the BDG field $Y_r$
  uniquely parameterises the complex $\alpha/2$-stable distribution
  $P^{\mu^z_r}$ via the scalar $\widehat{X}_r$; $\mu^z_r$ is its
  pushforward through $\Sigma \mapsto -1/(z - \Sigma)$; complex moments
  $\mathbb{E}\,G^q$ are single Laguerre integrals $g_{\alpha, 2q}(Y_r)$
  via Belinschi `eq:gziden`.
- **To be derived (9A.2 and 9B.2)**: closed-form reductions of
  $\mathbb{E}|G|^q$ and $\mathbb{E}(-\mathrm{Im}\,G)^q$ for $q \ne 1$.
  Both candidates may or may not collapse to single Laguerre integrals;
  if not, the 2-D pushforward integral against the explicit stable
  density is the fallback.
- **To be verified (9C)**: internal consistency at $q = 1$, tail-exponent
  matching, constant-$c$ reduction.
- **Implemented in `RMT/localisation.py`**:
  - `localisation_index_curve(...)` -- profile-aligned $\ell_q$
    (Part 1 of `localisation.py`, validated to machine precision at
    constant $c$).
  - `empirical_localisation_index_from_svd(...)` -- finite-$N$
    cross-check.
- **Not implemented: $D_q$ via the 2-D pushforward.** A prototype was
  drafted (`theoretical_Dq_curve`, `_density_sigma_via_ifft`) and
  then retracted. The 2-D pushforward needs the full 2-D distribution
  of $\Sigma$ on $\mathbb{C}^-$, which Belinschi's 1-scalar
  $\widehat{X}_r$ does not determine -- see sec. 9F below for the
  obstruction.
- **To be done (9D)**: select a path forward from the list in sec. 9F
  (rotationally-symmetric ansatz, population dynamics, additional
  self-consistency, empirical-only).

## 9F. 2-D pushforward: attempted, retracted

**Status: not viable from $Y_r$ alone.** This section was originally
written as the numerical fallback after the single-Laguerre reductions
failed (9A.2, 9B.2). A prototype implementation was drafted in
`localisation.py` and then removed because of the obstruction
documented below.

**The obstruction.** Belinschi's `cocott` formula
$$
\int e^{-itx}\, dP^\nu(x) = \exp\!\big(-\Gamma(1-\alpha/2)(it)^{\alpha/2}\,
\textstyle\int x^{\alpha/2}\, d\nu\big)
$$
parameterises $P^\nu$ through the **single complex scalar**
$X_r = \int x^{\alpha/2}\,d\nu$. This determines the LT/CF only along
a one-parameter slice of the 2-D Fourier domain (the line
$(\tau_1, \tau_2) = (-t, -it)$ for real $t > 0$). But BAG's general
definition of a complex $\alpha/2$-stable on $\mathbb{C}$ requires
**two scalar-valued functions** $\sigma_{\mu,\alpha/2}(t)$ and
$\beta_{\mu, \alpha/2}(t)$ of the direction of $t$ (BAG lines 1384-1394
of the preprint, eqs. (sigma), (beta)). The Belinschi 1-scalar formula
gives one slice; it does not determine $\sigma, \beta$ as functions of
direction, and so does not pin down the full 2-D distribution.

**BAG/Belinschi explicitly do not establish uniqueness** of the full
$\mu^z_r$ from $Y_r$. BAG line 405-407: "we cannot prove uniqueness
of the solution to this equation [for $\mu^z$]; we manage to prove
the uniqueness of the solution for $\int x^{\alpha/2}\,d\mu^z$" -- the
$Y_r$ scalar, no more.

**Numerical symptom of the obstruction.** The retracted prototype did
2-D inverse FFT of the 1-slice CF, treating it as if it were the full
2-D CF. Symptom: $\sim 37\%$ of the resulting "density" mass sits on
$\mathrm{Im}\,\Sigma > 0$, contradicting the $\mathbb{C}^-$ support
property of $\Sigma$ (BAG `cocott` requires support in
$\overline{\mathbb{C}^-}$). The 1-scalar CF is *not* the full 2-D CF,
and the inverse-FFT density is correspondingly wrong.

**Paths forward.** Selecting one requires user input:

1. *Closed-form analytic-only.* Restrict to analytic moments
   $\mathbb{E}[G^q] = g_{\alpha, 2q}(Y_r) \cdot \text{const}$
   (Belinschi `eq:gziden`). These are not the IPR moments
   ($\mathbb{E}[|G|^q]$, $\mathbb{E}[(\mathrm{Im}\,G)^q]$) for $q \ne 1$,
   so this approach delivers the spectral density (sec. 1) but not
   $D_q$.

2. *Rotationally-symmetric ansatz.* Assume $\mu^z_r$ has a known
   shape (e.g., complex $\alpha/2$-stable with $\beta = 0$ and one-
   parameter $\sigma$) which makes the 1-scalar Belinschi CF
   sufficient. Non-rigorous; consistent with Cizeau-Bouchaud sec. 7
   physics; user has expressed doubt about Cizeau-Bouchaud's
   derivation.

3. *Population dynamics.* Run the cavity RDE
   (`RMT.py:cavity_svd_resolvent`) to convergence, average
   $|G|^q$ or $(-\mathrm{Im}\,G)^q$ across the pool. Valid for
   moments that are finite in the wide-$N$ limit (convergent
   regime, $q < q_c$); for $q > q_c$ the pool-extreme statistics do
   not extrapolate to physical $N$ -- the user's stated objection
   applies in this regime.

4. *Additional self-consistency equations.* Beyond the BAG-Belinschi
   scalar $X_r$, derive new self-consistency equations for
   $\int |x|^{\alpha/2}\,d\mu^z$, $\int |x|^q\,d\mu^z$, etc. This is
   new derivation work not present in the BAG/Belinschi literature
   we have notes for.

5. *Empirical-only.* Drop theoretical $D_q$; harvest empirical $D_2$
   from `RMT.MLP_agg`'s aggregated singular-vector multifractal
   dimensions; plot. Phenomenological.

The status of the 2-D pushforward and the choice among (1)-(5) is the
content of any future revision of section 9D.

## 9G. One-ray ansatz: attempted (path 2), retracted

Path (2) of sec. 9F was attempted: ansatz $\Sigma = c_R\, R\, e^{i\theta_0}$
with $R$ one-sided $(\alpha/2)$-stable (Kanter LT convention
$\mathbb{E}[e^{-tR}] = e^{-t^{\alpha/2}}$), scale and phase calibrated to
Belinschi's `cocott` formula:
$$
c_R = (\Gamma(1-\alpha/2)\,|X_r|)^{2/\alpha}, \qquad
\theta_0 = (2/\alpha)\,\arg X_r.
$$

**Result: fails BDG self-consistency.** At $\alpha = 1.5$,
$z = 1.66 + 0.001\,i$ (a representative bulk point of an unstructured
$c \equiv 1$ test):

| | target (BDG $X_r$) | empirical from ansatz |
|---|---|---|
| $\mathbb{E}[G^{\alpha/2}]$ | $-1.35 - 1.97\,i$ | $0.11 - 0.018\,i$ |
| $\mathbb{E}[-\mathrm{Im}\,G]$ | $\pi \rho_{\text{base}} = 1.27$ | $0.012$ |

Magnitudes off by $\sim 20$; phases misaligned. The mismatch is not a
small numerical error -- the one-ray hypothesis is too restrictive to
capture the actual angular dispersion of $\mu^z_r$ on $\mathbb{C}^-$.
Belinschi's `cocott` formula calibrates the LT/CF along a single complex
line, which is structurally consistent with many distributions on
$\mathbb{C}^-$; the one-ray choice is the simplest but not the right
one.

**The ansatz code was removed** (`RMT/localisation.py` Part 2 placeholder
preserved). The empirical $D_q$ harvest in
`ht_mlp_jacobian.py:mlp_jacobian_Dq_spectrum` and notebook section 5
stands as the operational deliverable from path (5).

**Implications for the rest of path-list (sec. 9F):**

- Paths (1) and (5) remain valid and partially / fully implemented.
- Path (2) is closed for the one-ray sub-case. A richer ansatz (e.g.,
  two-parameter family with both radial scale and angular spread, or
  ansatz directly on $G$ rather than on $\Sigma$) could be attempted.
- Path (3) (population dynamics) and path (4) (additional self-
  consistency equations) remain open paths that have not been attempted
  yet.

## 9H. Analytical $D_q$ in terms of $G$, numerically realised via BAG cavity population dynamics

This section commits to the analytical expression for $D_q$ as a
functional of $\mu^z$ (the BAG/Belinschi cavity-resolvent distribution),
and verifies it numerically by realising $\mu^z$ via the cavity RDE.

### 9H.1 The spectral identity

For unit-normalised eigenvectors $\psi_k$ of the bipartite
Hermitisation $H$ with eigenvalues $\lambda_k$, the wide-$N$
self-averaging Lehmann identity is

$$
|\psi_\lambda(i)|^2 \;\stackrel{d}{=}\; \frac{-\mathrm{Im}\,G_{ii}(\lambda + i 0^+)}{N\, \pi\, \rho(\lambda)},
\qquad (12)
$$

with $G(z) = (H - z)^{-1}$ and $-\mathrm{Im}\,G_{ii} \ge 0$ for $z \in \mathbb{C}^+$.
The site-average of $-\mathrm{Im}\,G_{ii}$ recovers the spectral density:
$\rho(\lambda) = -\mathbb{E}_i\,\mathrm{Im}\,G_{ii}(\lambda + i 0^+)/\pi$.

### 9H.2 IPR moment in terms of $\mathrm{Im}\,G$

The inverse participation ratio
$I_q(\lambda) := \sum_i |\psi_\lambda(i)|^{2q}$ is, by (12) and wide-$N$
self-averaging,

$$
I_q(\lambda)
\;=\; \frac{1}{(N\pi\rho(\lambda))^q}\, \sum_i \big(-\mathrm{Im}\,G_{ii}\big)^q
\;\;\xrightarrow{N \to \infty}\;\;
\frac{N^{1-q}}{(\pi\rho(\lambda))^q}\, M_q(\lambda),
$$

where the **cavity-ensemble moment** is

$$
\boxed{\quad M_q(\lambda) \;:=\; \mathbb{E}_{\mu^z}\!\big[(-\mathrm{Im}\,G)^q\big],
\qquad z = \lambda + i 0^+. \quad} \qquad (13)
$$

### 9H.3 The multifractal exponent $D_q$

By the standard definition $I_q \sim N^{-(q-1) D_q}$:

$$
\boxed{\quad
D_q(\lambda) \;=\; -\frac{\log I_q(\lambda)}{(q-1)\,\log N}
\;=\; 1 \;-\; \frac{1}{(q-1)\,\log N}\, \log\!\bigg[\frac{M_q(\lambda)}{(\pi\rho(\lambda))^q}\bigg].
\quad} \qquad (14)
$$

The formula depends on $\mu^z$ only through the **single scalar functional**
$M_q/(\pi\rho)^q$ -- a ratio of two cavity-ensemble moments. The $N$
appearing in (14) is the **physical** matrix size, not a pool size, so
this gives a finite-$N$ prediction directly comparable to empirical SVD.

### 9H.4 BAG/Belinschi 2-D stable input for $M_q$

By BAG `theo-limitpoint` + Belinschi `cocott`, $\mu^z$ is the pushforward
of the complex $\alpha/2$-stable distribution $P^{\mu^z}$ on $\mathbb{C}^-$
under the rational map $\Sigma \mapsto -1/(z - \Sigma)$, with
$P^{\mu^z}$ characterised by BAG's $\sigma_{\mu, \alpha/2}(t),
\beta_{\mu, \alpha/2}(t)$ (preprint lines 1384-1408). The self-consistency
$\mu^z = (\Sigma \mapsto -1/(z-\Sigma))_\ast P^{\mu^z}$ is what the cavity
RDE solves.

Sampling from $P^{\mu^z}$ analytically requires the full $\sigma, \beta$
direction-functions, which BAG do not pin down from $Y_r$ alone (see
9F). Instead we realise $\mu^z$ **distributionally** via the cavity RDE:
maintain a finite-size pool of $G$ samples and iterate the cavity
recursion (which IS the BAG pushforward operator $\mu \mapsto
(\Sigma \mapsto -1/(z-\Sigma))_\ast P^{\mu}$, applied as a sample-update
in the wide-pool limit). At convergence, the pool is a Monte-Carlo
sample of $\mu^z$.

$M_q$ from the pool is the empirical $(1/P)\sum_k (-\mathrm{Im}\,g_k)^q$.
For $q < q_c$ (convergent regime where the limit moment is finite),
this converges to the BAG-prediction $M_q$ as pool size grows; for
$q > q_c$ the cavity-ensemble moment is infinite in the limit and the
pool-average gives a pool-size-dependent estimate. The latter regime is
where the user's earlier objection to population dynamics applies; for
the former regime the population-dynamics estimate IS the correct BAG
moment.

### 9H.5 Numerical implementation

`RMT.py:cavity_svd_resolvent` runs the cavity RDE for the bipartite
Hermitisation. It maintains two populations: $g_1$ (row-side
$\mu^z_r$) and $g_2$ (column-side $\mu^z_c$). For one-sided
$|\tau|=c(y)$ structured matrices, the $\chi$-samples carry the
column profile via $\chi_j = \sigma_w\,|\phi'((q^*)^{1/\alpha} z_j)|$
with $z_j \sim p_\alpha$ (cf. `RMT.py:jac_cavity_svd_log_pdf`).

From the converged populations:

- $M_q^{(1)}(\lambda) = (1/P)\sum_k (-\mathrm{Im}\,g_{1,k})^q$ for the
  row side (left singular vectors).
- $M_q^{(2)}(\lambda)$ same for $g_2$ (column side / right vectors).
- $\rho(\lambda) = -\mathbb{E}_\text{pool}\,(\mathrm{Im}\,g_1 +
  \mathrm{Im}\,g_2)/(2\pi)$ (the bipartite resolvent gives both halves).

Then $D_q$ via (14) for each side. The implementation is
`localisation.theoretical_Dq_curve_popdyn(...)`.

**Validation expectation.** At $1 < \alpha < 2$, BG prove $D_q \to 1$
as $N \to \infty$ for the unstructured case. At finite $N$, $D_q < 1$
with the $\log N$-rate of convergence given by (14). For the structured
case (one-sided $c(y)$), $D_q^{(1)}, D_q^{(2)}$ should approach 1
asymptotically with $N$, with possibly different finite-$N$ corrections
on row vs column sides due to the column-profile-induced asymmetry of
$\mu^z_c(y)$. Comparison to `mlp_jacobian_Dq_spectrum` at the same
$(\alpha, \sigma_w, N)$ tests both: that the BDG matrix ensemble is the
right model AND that the cavity-RDE-extracted $M_q$ correctly predicts
finite-$N$ multifractality.

## 9I. Towards closed-form $M_q$: self-consistency on the spectral measure $\Gamma$

The premise of sec. 9F was wrong in framing: it claimed Belinschi/BAG
"cannot determine $\mu^z$" from $Y_r$. The correct statement is that
**BAG prove uniqueness only at the level of $X_r$** (the $\alpha/2$-th
moment) and **do not** prove uniqueness of the full $\mu^z$. Whether
$\mu^z$ is in fact determined by $X_r$ alone (i.e., whether the BDG
self-consistency closes in one scalar) is an open analytical question.
This section writes out the self-consistency equation for the full
distributional structure -- specifically the spectral measure $\Gamma$
of $P^{\mu^z}$ on $S^1$ -- as a starting point for the analytical
attack.

### 9I.1 The spectral-measure parameterisation

By the standard structural theorem for complex $\alpha/2$-stable
distributions on $\mathbb{C}$ (Samorodnitsky-Taqqu), $P^{\mu^z}$ is
uniquely determined by a **spectral measure** $\Gamma$ on the unit
circle $S^1$, related to BAG's $\sigma, \beta$ via

$$
\sigma_{\mu, \alpha/2}(\hat t)^{\alpha/2} = \frac{1}{C_{\alpha/2}} \int_{S^1} |\langle \hat t, \hat s\rangle|^{\alpha/2}\, \Gamma(d\hat s),
\qquad
\beta_{\mu, \alpha/2}(\hat t) = \frac{\int_{S^1} |\langle\hat t, \hat s\rangle|^{\alpha/2}\,\mathrm{sign}\langle\hat t, \hat s\rangle\, \Gamma(d\hat s)}{\int_{S^1} |\langle\hat t, \hat s\rangle|^{\alpha/2}\, \Gamma(d\hat s)}.
$$

$\Gamma$ is invariant under $\hat s \mapsto -\hat s$ if $P^\mu$ is
symmetric, or has a specific asymmetry encoded in $\beta$.

### 9I.2 BDG self-consistency for $\Gamma$

The cavity-RDE fixed point is

$$
\mu^z \;=\; \big(\Sigma \mapsto -1/(z - \Sigma)\big)_\ast\, P^{\mu^z}.
$$

At the level of $\Gamma$ (equivalently $\sigma$, $\beta$), this gives a
**functional fixed-point equation**: for every $\hat t \in S^1$,

$$
\boxed{\quad
\int_{S^1} |\langle\hat t, \hat s\rangle|^{\alpha/2}\, \Gamma(d\hat s)
\;=\; C_{\alpha/2}\, \int_{\mathbb{C}^-} \Big|\Big\langle\hat t,\, -\frac{1}{z - \Sigma}\Big\rangle\Big|^{\alpha/2}\, dP^{\Gamma}(\Sigma),
\quad} \qquad (15)
$$

with the analogous signed-version equation for $\beta$. The RHS depends
on $\Gamma$ through $dP^{\Gamma}$; the LHS is a directional fractional
moment along $\hat t$. This is a self-consistency equation in $\Gamma$,
on the function space of probability measures on $S^1$.

In terms of the **scale-weighted angular density** of $\mu$ on
$\mathbb{C}^-$,

$$
\tilde h_\mu(\phi) \;:=\; \int_0^\infty r^{\alpha/2}\, f_\mu(r e^{i\phi})\, r\, dr,
\qquad \phi \in [-\pi, 0],
$$

(where $f_\mu$ is the 2-D density of $\mu^z$), eq. (15) becomes

$$
\frac{1}{C_{\alpha/2}}\, \int_{-\pi}^{0} |\cos(\phi - \theta)|^{\alpha/2}\, \tilde h_\mu(\phi)\, d\phi
\;=\; \mathrm{[RHS\ in\ terms\ of\ pushforward\ of\ } P^\Gamma\mathrm{]},
\qquad (16)
$$

i.e. the LHS is a convolution of $\tilde h_\mu$ with the kernel
$|\cos|^{\alpha/2}$ on $S^1$. The cavity self-consistency closes if the
pushforward map sends this convolution back to itself in a structured way.

### 9I.3 Fourier expansion ansatz

Parameterise $\tilde h_\mu(\phi)$ on $[-\pi, 0]$ in Fourier modes (real
basis, using $\sin, \cos$):

$$
\tilde h_\mu(\phi) \;=\; \frac{a_0}{2} + \sum_{n=1}^\infty \big[a_n \cos(2 n \phi + \pi n) + b_n \sin(2 n \phi + \pi n)\big],
$$

(half-circle Fourier basis on $[-\pi, 0]$; the exact basis choice is
a parameterisation choice). Plugging into (15) gives a hierarchy of
equations on the Fourier coefficients $\{a_n, b_n\}_{n \ge 0}$.

**The key open question:** does the BDG self-consistency hierarchy
**truncate at finite Fourier order** at the fixed point? If yes:
$\Gamma$ has a parametric closed form, and so does $M_q$ for any $q$
(integer or non-integer), and hence the closed-form $D_q$ theory is
achievable.

If no (the hierarchy stays infinite-dimensional), the BDG fixed-point
$\mu^z$ is genuinely an infinite-dimensional object and closed-form
$M_q$ requires either ansatz closures (path 2 of sec. 9F) or
distributional methods (path 3).

### 9I.4 The single-scalar anchor $X_r$ as a check

Belinschi's `cocott` gives the constraint:

$$
\int_{-\pi}^0 e^{i (\alpha/2) \phi}\, \tilde h_\mu(\phi)\, d\phi \;\propto\; X_r(z),
$$

i.e. the **$(\alpha/2)$-th Fourier mode** of $\tilde h_\mu$ (in the
complex-exponential basis) equals the scalar $X_r$ up to known prefactors.
This is one complex equation = two real constraints, fixing $a_{\alpha/4}$
(modulo basis conventions). The Fourier hierarchy of sec. 9I.3 must be
consistent with this, providing one anchor.

### 9I.5 Concrete next steps

To attack the closed-form question:

1. **Expand the RHS of (15)** in Fourier modes of $\tilde h_{P^\mu}(\psi)$
   (the angular weight of $\Sigma$, related to $\Gamma$). This requires
   evaluating the pushforward integral $\int |1/(z - \Sigma)|^{\alpha/2}
   \cos(\arg(-1/(z-\Sigma)) - \theta) \cdot \text{etc}\, dP^\Gamma$ in
   polar coordinates of $\Sigma$.
2. **Project onto Fourier modes** of $\tilde h_\mu(\phi)$.
3. **Check truncation**: do mode $n$ equations involve only modes
   $\le n + n_0$ for some finite $n_0$? If yes, the system closes after
   $n_0$ modes.
4. **If closed**: solve the truncated system, get $\tilde h_\mu$ in
   closed form, compute $M_q = \int_0^\infty r^q d\mu^z(re^{i\phi})$
   pointwise.
5. **If not closed**: identify the structural reason and decide
   between distributional realisation (population dynamics) or a
   structural ansatz.

This is non-trivial analytic work but well-defined. The key technical
step is expanding the pushforward integral in mode-coupling form;
nothing in the framework is obstructive a priori.

### 9I.6 Status and recommendation

- **Open analytical question**: does the BDG self-consistency on $\Gamma$
  close in finite Fourier modes?
- **Numerical probe (much cheaper)**: sample $\mu^z$ via cavity
  population dynamics, compute $\sigma_\mu(\hat t)^{\alpha/2}$ for many
  directions $\hat t \in S^1$, check whether $\sigma_\mu$ has a low-rank
  Fourier representation. This informs whether the analytical attack
  is likely to bear fruit.
- **Status of "Belinschi can't give $M_q$" claim from earlier sections**:
  RETRACTED. The correct statement is that Belinschi gives only the
  $X_r$ anchor; the full $\Gamma$ may or may not close in finite
  parameters at the BDG fixed point, and that's the open analytical
  question of this section.

## 9J. Density-deviation diagnostic: practical localisation onset

A practical alternative to the full $D_q$ / $\Gamma$ derivation: detect
localisation onset by **deviation between two estimators of the same
first moment**.

### 9J.1 Principle

The spectral density $\rho(s)$ is the first moment
$\rho = -\mathbb{E}_{\mu^z}\,\mathrm{Im}\,G/\pi$. Two ways to estimate it:

- **Analytical**: $\rho_\text{thy}(s) = -\mathrm{Im}\,h_\alpha(Y_r(s + i 0^+))/(\pi s)$
  via the deterministic BDG field
  (`one_sided_wishart_levy.theoretical_one_sided_singular_value_curve`).
- **Pool-mean**: $\rho_\text{popdyn}(s) = \mathbb{E}_\text{pool}[-\mathrm{Im}\,g]/\pi$
  via the cavity-RDE population dynamics
  (`RMT.cavity_svd_resolvent` / `RMT.jac_cavity_svd_log_pdf`).

**In the delocalised regime**: the local resolvent $\mathrm{Im}\,g_{ii}$
across the pool has bounded variance, pool-mean converges to its limit at
$P^{-1/2}$. Popdyn and analytical agree to within MC noise.

**In the localised regime**: $\mathrm{Im}\,g_{ii}$ across the pool is
heavy-tailed (Aizenman-Molchanov / BG mechanism), pool-mean has slow
convergence dominated by pool extremes. **Popdyn systematically deviates
from analytical** at any fixed pool size, with deviation magnitude
quantifying the heavy-tail mechanism.

This is the Aizenman-Molchanov fractional-moment criterion re-expressed
at the density level: the localisation signature lives in the *failure*
of the wide-pool / wide-$N$ self-averaging assumption that the
analytical theory uses.

### 9J.2 Hermitisation-driven $\alpha$ mapping

For an $\alpha$-stable rectangular matrix, the bipartite Hermitisation
cavity self-energy involves *squared* heavy-tailed entries, halving the
stability index. So the bipartite Hermitisation's effective
Wigner-Levy stability is $\alpha_\text{Wigner-equiv} = \alpha/2$, and
the BG regimes map:

| $\alpha_\text{SV}$ | $\alpha_\text{Wigner-equiv}$ | BG regime |
|---|---|---|
| $> 4/3$ | $> 2/3$ | open intermediate (BG sec. 1) |
| $< 4/3$ | $< 2/3$ | proven localised in heavy-tail spectrum |

The operational regime $\alpha_\text{SV} \in (1, 2)$ for the heavy-tailed
MLP Jacobian therefore **straddles BG's localisation transition** at
$\alpha_\text{SV} = 4/3$.

### 9J.3 Concrete diagnostic and empirical findings

Implementation: `ht_mlp_jacobian.density_deviation_diagnostic`. Run at
fixed pool size $P = 2^{n_\text{doublings}}$, multiple $\chi$ realisations
for cross-sample noise estimation. Output: $|\rho_\text{popdyn}(s) -
\rho_\text{thy}(s)|$ vs $s$, with the cross-$\chi$ standard deviation as
the MC noise floor for signal-to-noise interpretation.

Empirical findings at $\sigma_w = 1$, $n_\text{doublings} = 7$,
num_chis = 8 (3 $\alpha$ values):

| $\alpha_\text{SV}$ | $\alpha_\text{Wigner-equiv}$ | BG regime | max $\|\Delta_\rho\|$ | argmax $s$ |
|---|---|---|---|---|
| 1.2 | 0.60 | proven localised | **0.117** | 1.6 |
| 1.5 | 0.75 | open intermediate | 0.036 | 2.4 |

Max deviation is 3x larger at $\alpha = 1.2$ than at $\alpha = 1.5$,
exactly the trend predicted by Hermitisation + BG. Strongest deviation
is in the spectral tail/cross-over region (not the bulk), consistent
with the standard heavy-tailed localisation picture (eigenvectors at
large eigenvalues are localised first).

### 9J.4 What the diagnostic does and doesn't deliver

**Delivers:**
- Quantitative localisation-onset $s_c$ as the SV where systematic
  deviation exceeds MC noise.
- $\alpha$-trend matching the Hermitisation-mapped BG regimes.
- A pool-size sweep gives the heavy-tail exponent $\nu$ of the local
  resolvent distribution (the multifractal-spectrum diagnostic) by
  fitting $|\Delta_\rho(s; P)|$ vs $P$ -- still TODO.

**Does not deliver:**
- A closed-form $D_q$ value at given $s$. The diagnostic is qualitative
  ("localisation onset detected") and quantitative-as-deviation-magnitude,
  but doesn't predict $D_q$.
- An analytical theory of $\Gamma$. The diagnostic establishes that the
  BDG framework breaks down at some $s_c$, motivating but not replacing
  sec. 9I's analytical attack.

### 9J.5 Next refinements

- **Pool-size scaling sweep** $(P \to 2P)$ -- exponent of $|\Delta_\rho(s; P)|$
  as $P \to \infty$ gives the heavy-tail index. Cheap to implement on
  top of the existing diagnostic.
- **Higher $\alpha$-resolution sweep** to identify a sharper $\alpha_c$
  transition in the operational range.
- **Independent verification** against MLP-Jacobian SVD localisation at
  finite $N$: at SVs above $s_c$, the eigenvector IPRs should show
  $D_q < 1$ trends consistent with the diagnostic.

## 9K. Pool-size scaling: heavy-tail index $\nu(s)$ from finite-pool deviation

The single-pool-size diagnostic of 9J gives a binary "deviation vs no
deviation" picture. Sweeping the pool size $P = 2^{n_d}$ gives a
**quantitative** heavy-tail index $\nu(s)$ at each spectral position.

### 9K.1 Generalised-CLT scaling

For an iid sample of size $P$ from a distribution with tail
$\Pr(|X| > u) \sim u^{-\nu}$:

| $\nu$ regime | mean and variance | pool-mean deviation scaling |
|---|---|---|
| $\nu > 2$ | finite mean and var | CLT, $\|\hat\mu - \mu\| \sim P^{-1/2}$ |
| $1 < \nu < 2$ | finite mean, infinite var | heavy-tail CLT, $\|\hat\mu - \mu\| \sim P^{1/\nu - 1}$ |
| $\nu = 1$ | mean diverges marginally (log) | $\|\hat\mu - \mu\| \sim 1$ (constant in $P$) |
| $\nu < 1$ | infinite mean | $\|\hat\mu - \mu\| \sim P^{1/\nu - 1}$ grows with $P$ |

Fitting the slope $m$ of $\log|\Delta_\rho(s; P)|$ vs $\log P$ at each
$s$, the heavy-tail index of the local resolvent distribution at SV $s$
is

$$
\boxed{\quad \nu(s) \;=\; \frac{1}{m(s) + 1}. \quad} \qquad (17)
$$

Map:

- $m = -1/2 \;\Rightarrow\; \nu = 2$: CLT (delocalised).
- $-1/2 < m < 0 \;\Rightarrow\; 1 < \nu < 2$: heavy-tail-CLT regime,
  partial localisation, pool-mean still converges but slowly.
- $m = 0 \;\Rightarrow\; \nu = 1$: marginal -- pool-mean doesn't
  converge; **localisation onset**.
- $m > 0 \;\Rightarrow\; \nu < 1$: pool-mean diverges with $P$ ---
  strong localisation.

### 9K.2 Empirical findings at $\alpha = 1.5$, $\sigma_w = 1$

At pool sizes $P \in \{16, 32, 64, 128, 256\}$, num_chis$=8$:

- *Bulk* ($s \in [0.2, 1.2]$): mean slope $\approx -0.68$ (faster than
  CLT). The faster-than-CLT scaling is the cavity-iteration convergence
  rate (`cavity_svd_resolvent` does $P^2$ iterations per doubling), not
  pure independent-sample statistics. **No localisation signal here**.

- *Mid* ($s \in [1.5, 2.5]$): slopes scatter; transition region.

- *Tail* ($s \in [2.5, 4.0]$): slope converges to $\approx 0$
  ($\nu \to 1$). $|\Delta_\rho|$ becomes **flat in $P$** at large $s$:
  the pool-mean fails to converge as the pool grows. This is the
  heavy-tail signature -- **localisation onset $s_c \approx 2.0$-$2.5$
  for these parameters**.

### 9K.3 Caveat on bulk slopes (and the decoupled-iteration fix)

The faster-than-CLT scaling in the bulk is *not* a delocalisation
signature -- it's the iteration-convergence rate of `cavity_svd_resolvent`,
which by default uses `num_steps = P^2` per doubling. As $P$ grows,
iterations-per-element scales as $P$, so the cavity dynamics converges
to its fixed point super-CLT. This dominates over pool-mean MC noise at
small to moderate pool sizes. The diagnostic signature is therefore the
*crossover* from fast convergence (bulk, $m \ll -1/2$) to flat ($m = 0$,
tail), not the absolute slope in any region.

**Fix applied:**
`density_deviation_pool_sweep(num_steps_per_element=100)` passes
`num_steps_fn = lambda P: 100 * P` to popdyn, so each pool element is
updated $\sim 100$ times *regardless of pool size*. This decouples the
mixing (iterations-per-element) from the pool-mean MC noise.

With the fix, at $\alpha_\text{SV} = 1.5$ the picture is sharp:

- Bulk ($s < 1$): mean slope $\approx -0.8$ (still faster than CLT --
  pool elements share the $\chi$-profile so are not iid, but decay
  is clear).
- Tail ($s > 1$): slope $\equiv 0$ exactly across every SV in the
  tail. $|\Delta_\rho|$ is flat in $P$ -- the strongest possible
  localisation signature.
- **Localisation onset $s_c \approx 1.0$** at $(\alpha_\text{SV}, \sigma_w)
  = (1.5, 1.0)$.

This is consistent with Tarquini-Biroli-Tarzia 2016 (TBT,
`.agents/notes/tarquini-2015.md`) at $\mu_\text{TBT} = \alpha_\text{SV}/2
= 0.75$: their mobility edge equation (TBT eq:mobility) predicts an
$E^\star$ in their localised regime $\mu < 1$, and the Hermitisation
correspondence identifies this with our $s_c$ up to a normalisation
factor.

### 9K.4 Localisation-onset $s_c$ as a function of $\alpha$

Run sweep at multiple $\alpha$ to map the localisation onset across the
operational range. Prediction from Hermitisation + BG:

- $\alpha_\text{SV} \in (1, 4/3)$: strong tail localisation,
  $s_c$ relatively small (early onset).
- $\alpha_\text{SV} \in (4/3, 2)$: weaker localisation, $s_c$ closer to
  spectral edge.
- $\alpha_\text{SV} \to 2$: $s_c \to \infty$ (no localisation,
  Marchenko-Pastur limit).

This $\alpha$-sweep + $s_c$-mapping is the natural quantitative output
of the diagnostic.

