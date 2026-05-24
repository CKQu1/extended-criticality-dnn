# Heavy-tailed extension of MLP mean-field theory

This file develops a heavy-tailed analogue of the random-MLP mean-field
theory summarised in `.agents/notes/light-tailed-mlp.md`. Section 1 sets up
the model and fixes parameterisation conventions chosen so that the
$\alpha = 2$ limit reproduces Poole et al. (2016)'s Gaussian length map
verbatim. Section 2 derives the alpha-length map by applying the
$\alpha$-stable CLT to the infinite-width per-neuron pre-activation.
Section 3 checks the $\alpha = 2$ reduction. Section 4 catalogues the
structural features that distinguish the heavy-tailed case from the
Gaussian case -- chiefly: the marginal of $h^l_j$ is $\alpha$-stable
rather than Gaussian, and the natural order parameter $q^l$ is the
$\alpha$-th moment of $\phi(h^l)$ rather than the second moment.

Section 5 derives the bivariate-stable closure (the $\mathcal{C}$-map
analogue) and shows it is parameterised by an infinite-dimensional
spectral measure on $S^1$ for $\alpha < 2$, not a scalar correlation.
Section 6 derives the exact unconditional perturbation recursion (eq.
10) and its small-perturbation linearisation (eq. 11), then notes that
the natural Poole-style scalar closure of eq. 11 would assume a
mean-field decoupling that direct simulation (`heavy_tailed_mlp.ipynb`,
Test 1) shows fails for $\alpha < 2$; the operational edge of chaos is
therefore read off from the Lyapunov exponent of the nonlinear forward
dynamics, which Test 4 measures across $(\alpha, \sigma_w)$. Section 7
checks that the framework reduces to Poole's exactly at $\alpha = 2$
(where the decoupling is rigorous). Section 8 is a structural appendix:
it names the spectral-measure state space, verifies the fully-correlated
fixed point, and records an auxiliary operator-level diagonal entry
$L_{q_{12}, q_{12}}$ that doubles as a numerical sanity check on the
bivariate-stable closure. Backward / gradient analysis and the
Jacobian-spectrum link to the structured Wishart-Levy theory of
`RMT/structured_wishart_levy.md` are deferred to follow-on derivations.

## 1. Setup

A depth-$L$ fully-connected MLP with layer widths $N_l$ (treated as
constant $N$ throughout for brevity; the heterogeneous case rescales by
$N_{l-1}^{-1/\alpha}$ per layer in the obvious way). Pre-activations
propagate by

$$
h^l_i = \sum_{j=1}^N W^l_{ij}\, \phi(h^{l-1}_j) + b^l_i,
\qquad i = 1, \dots, N,
\qquad l = 1, \dots, L,
$$

with i.i.d. weights $W^l_{ij}$. We set $b^l_i \equiv 0$ for clarity in
Sections 2-3; biases re-enter additively in Section 3 eq. (7).

The weight law is **symmetric $\alpha$-stable**, $\alpha \in (0, 2]$,
parameterised so that the $\alpha = 2$ limit recovers a standard Gaussian:

$$
W^l_{ij} = \sigma_w\, N^{-1/\alpha}\, \tilde W^l_{ij},
\qquad
\tilde W \sim p_\alpha,
\qquad
\hat p_\alpha(t) := \mathbb{E} e^{it \tilde W} = \exp\!\big(-\tfrac{1}{2} |t|^\alpha\big).
$$

For $\alpha = 2$ this gives $\tilde W \sim \mathcal{N}(0, 1)$ and
$\mathrm{Var}(W^l_{ij}) = \sigma_w^2 / N$ -- the standard Poole et al.
convention. For $\alpha \in (0, 2)$, $\tilde W$ has no second moment; the
natural scale-of-magnitude is the BDG quantile $a_N$, which under this
normalisation is $\Theta(\sigma_w\, N^{-1/\alpha})$, the same order as the
Gaussian case.

The pointwise nonlinearity $\phi$ is assumed **bounded** with $\phi(0) = 0$
(e.g. $\tanh$, soft-sign). Unbounded $\phi$ (ReLU, linear) makes the length
map below diverge -- see Section 4(c).

## 2. Derivation of the alpha-length map

**Step 1: conditional distribution of $h^l_i$ given the previous layer.**
At fixed $h^{l-1}$ (and hence fixed activations $a_j := \phi(h^{l-1}_j)$),
the sum $h^l_i = \sum_j W^l_{ij}\, a_j$ is a linear combination of i.i.d.
symmetric $\alpha$-stable variables. The standard stability identity (which
follows by multiplying characteristic functions: $\prod_j \hat
p_\alpha(a_j t)$ collapses to $\hat p_\alpha((\sum_j |a_j|^\alpha)^{1/\alpha} t)$)
gives, in distribution,

$$
\sum_j W^l_{ij}\, a_j
\stackrel{d}{=}
\sigma_w\, N^{-1/\alpha} \left(\sum_j |a_j|^\alpha\right)^{1/\alpha}\, \tilde W
= \sigma_w \left(\frac{1}{N} \sum_j |\phi(h^{l-1}_j)|^\alpha\right)^{1/\alpha} \tilde W.
$$

Therefore $h^l_i \mid h^{l-1}$ is symmetric $\alpha$-stable with scale

$$
S^l := \sigma_w \left(\frac{1}{N} \sum_{j=1}^N |\phi(h^{l-1}_j)|^\alpha\right)^{1/\alpha}. \qquad (1)
$$

**Step 2: self-averaging of the empirical $\alpha$-th moment.** As
$N \to \infty$, the empirical average in (1) concentrates on its
expectation, provided $\mathbb{E}[|\phi(h^{l-1})|^\alpha]$ is finite (which
holds for any bounded $\phi$):

$$
\frac{1}{N} \sum_{j=1}^N |\phi(h^{l-1}_j)|^\alpha
\;\longrightarrow\;
\mathbb{E}\big[|\phi(h^{l-1})|^\alpha\big]. \qquad (2)
$$

The expectation in (2) is over $h^{l-1}_j$. By the same stability argument
applied recursively at layer $l-1$, the marginal of $h^{l-1}_j$ is itself
symmetric $\alpha$-stable with scale $S^{l-1}$: $h^{l-1}_j \stackrel{d}{=}
S^{l-1} z$ with $z \sim p_\alpha$.

**Step 3: change of variable.** Substituting $h^{l-1} = S^{l-1} z$ in (2),

$$
\mathbb{E}\big[|\phi(h^{l-1})|^\alpha\big]
= \int p_\alpha(z)\, |\phi(S^{l-1} z)|^\alpha\, dz.
$$

Combining with (1):

$$
S^l = \sigma_w \left( \int p_\alpha(z)\, |\phi(S^{l-1} z)|^\alpha\, dz \right)^{1/\alpha}. \qquad (3)
$$

Equation (3) now defines $S^l$ as the deterministic $N \to \infty$ limit of
the random scale (1). From here on $S^l$ denotes this limit.

**Step 4: the alpha-length map in scale-power form.** Define the natural
heavy-tailed length-like order parameter

$$
q^l := (S^l)^\alpha. \qquad (4)
$$

Raising (3) to the $\alpha$-th power and substituting (4) yields the
**heavy-tailed length map**

$$
\boxed{\quad
q^l \;=\; V_\alpha(q^{l-1})
\;:=\; \sigma_w^\alpha \int p_\alpha(z)\,
       \big|\phi\big( (q^{l-1})^{1/\alpha}\, z \big)\big|^\alpha\, dz.
\quad} \qquad (5)
$$

## 3. Reduction to the Gaussian length map at $\alpha = 2$

At $\alpha = 2$ the convention $\hat p_2(t) = \exp(-\tfrac{1}{2} t^2)$ gives
$\tilde W \sim \mathcal{N}(0, 1)$, so $p_2(z)\, dz = Dz$ (standard Gaussian
measure). Equation (5) becomes

$$
q^l = \sigma_w^2 \int Dz\, \phi\!\big(\sqrt{q^{l-1}}\, z\big)^2, \qquad (6)
$$

which is Poole et al. (2016) eq. (2) at $\sigma_b = 0$. Adding biases:
**additive symmetric $\alpha$-stable biases** $b^l_i = \sigma_b \tilde B^l_i$
with $\tilde B \sim p_\alpha$ independent of the weights. Adding (1)'s sum
and the bias term inside the $\alpha$-norm,

$$
(S^l)^\alpha = \sigma_w^\alpha \int p_\alpha(z) |\phi(S^{l-1} z)|^\alpha\, dz
             + \sigma_b^\alpha,
$$

i.e.

$$
q^l = V_\alpha(q^{l-1}) + \sigma_b^\alpha. \qquad (7)
$$

The bias contributes additively at the $\alpha$-power level, mirroring the
Gaussian case at the variance level (Poole et al. eq. (2) with $\sigma_b^2$).
At $\alpha = 2$, (7) reduces to $q^l = \sigma_w^2 \int Dz\, \phi(\sqrt{q^{l-1}} z)^2 + \sigma_b^2$
exactly.

## 4. Structural features and caveats

**(a) Order parameter is the $\alpha$-th moment, not the second moment.**
$q^l$ is naturally defined as $(S^l)^\alpha$ -- the $\alpha$-th power of the
$\alpha$-stable scale of $h^l$ -- not as a variance. The integrand in
$V_\alpha$ is $|\phi|^\alpha$, not $\phi^2$. At $\alpha = 2$ the two
collapse onto the same object; off $\alpha = 2$ they are genuinely
different, and identities that rely on Gaussian Stein-type integration by
parts (e.g. derivative tricks linking $\langle \phi^2 \rangle$ to
$\langle \phi'^2 \rangle$) need re-derivation.

**(b) Marginal of $h^l_j$ is $\alpha$-stable, not Gaussian.** The Gaussian
closure that underpins Poole's analysis is replaced by an $\alpha$-stable
closure. Every per-layer expectation against the marginal of $h$ becomes
an integral against $p_\alpha$. In particular, the two-input correlation
map (next derivation) involves a *bivariate* symmetric $\alpha$-stable
measure on $(h^{l-1}(x^1), h^{l-1}(x^2))$, which for $\alpha < 2$ is not
parameterised by a single correlation coefficient: jointly $\alpha$-stable
distributions live on a spectral measure on the unit circle, richer than
the scalar Gaussian covariance. The Poole $\mathcal{C}$-map analogue must
account for this richer dependence structure.

**(c) Bounded $\phi$ is required.** For unbounded $\phi$ -- e.g. ReLU,
linear -- the integral $\int p_\alpha(z) |\phi(S z)|^\alpha\, dz$ diverges:
$p_\alpha(z) \sim c_\alpha |z|^{-1-\alpha}$ as $|z| \to \infty$, and
$|\phi(Sz)|^\alpha$ grows at least as $S^\alpha |z|^\alpha$ on a half-line,
so the integrand is $\sim |z|^{-1}$ at infinity and diverges
logarithmically. The heavy tail of $p_\alpha$ is exactly enough to make
$|\phi|^\alpha$ non-integrable when $|\phi|$ is itself $\alpha$-tailed
unbounded. Hence the length-map order parameter $q^l$ is infinite for
unbounded nonlinearities, and the framework as stated only applies to
bounded $\phi$ ($\tanh$, sigmoid, soft-sign, hard-tanh, etc.). Extensions
to ReLU-type activations require either a heavy-tailed-friendly
re-parameterisation of the order parameter (e.g. tracking the spectral
measure of $h^l$ rather than a single scale), or a truncation cutoff.

**(d) Fixed-point structure: continuum vs finite-grid behaviour.**
For $\phi$ bounded with $\phi(0) = 0$ and $\phi'(0) = c \neq 0$,
$V_\alpha(0) = 0$ and $V_\alpha(q) \le \sigma_w^\alpha \|\phi\|_\infty^\alpha$.
Splitting the integrand at the saturation transition
$|z| \sim q^{-1/\alpha}$ and using the heavy tail
$p_\alpha(z) \sim c_\alpha |z|^{-1-\alpha}$ gives the near-origin asymptote

$$
V_\alpha(q) \sim
  \frac{\sigma_w^\alpha\, c^\alpha\, c_\alpha}{\alpha}\, q\, \log(1/q)
  \;+\; O(q) \qquad (q \to 0^+).
$$

The slope at the origin diverges, but only **logarithmically slowly**:
$V_\alpha(q)/q \to +\infty$ via $\log(1/q)$, not a power-law cusp. Two
practical consequences (both checked in `heavy_tailed_mlp.ipynb`,
Test 0b):

- *Continuum theory.* $V_\alpha$ leaves the diagonal above immediately,
  so a nontrivial $q^* > 0$ exists for every $\sigma_w > 0$ when
  $\alpha < 2$. Matching $\sigma_w^\alpha c_\alpha c^\alpha \alpha^{-1}
  \log(1/q^*) \sim 1$ gives $q^* \sim \exp(-\text{const}/\sigma_w^\alpha)$:
  exponentially small in $1/\sigma_w^\alpha$. The forward MLP with
  CMS-sampled untruncated weights sees this continuum behaviour --
  small-$\sigma_w$ networks have small but strictly positive activation
  scale, no spurious bifurcation.
- *Finite-grid numerics.* The analytical iteration `length_fixed_point`
  computes (5) on a discrete $z$-grid $|z| \le Z$. The truncation caps
  $V_\alpha(q)/q$ at $\sigma_w^\alpha M_\alpha(Z)$ where
  $M_\alpha(Z) \sim (2 c_\alpha/\alpha) \log Z$, reinstating a
  Gaussian-like threshold $\sigma_w^\alpha M_\alpha(Z) = 1$ below which
  the iteration converges to floating underflow rather than the
  (exponentially small) true $q^*$. With Test 0b's $Z = 80$,
  $M_{1.5} \approx 1.79$.

The mathematical "no $\sigma_w$-bifurcation" property is therefore real
for the physical network but operationally inaccessible from the
analytical side at small $\sigma_w$. Sections 5--7 should be read as
describing the regime $\sigma_w \gtrsim 1$ where $q^*$ is order unity
under both pathways. At $\alpha = 2$ everything collapses: $V_2'(0) =
\sigma_w^2 |\phi'(0)|^2$ is finite without any truncation and the
threshold $\sigma_w |\phi'(0)| = 1$ is the standard Poole length-map
bifurcation.

**(e) Connection to the random-matrix theory of `RMT/`.** The Jacobian
of an $\alpha$-stable MLP is a product of structured heavy-tailed random
matrices, whose limiting spectrum is governed by the structured
Wishart-Levy law of `RMT/structured_wishart_levy.md`. The column profile
$c(y) = |\phi'(h^{l-1}(y))|$ plays the role of the deterministic
column-scale $\tau(x, y) = c(y)$ in Theorem 2 of that derivation. The
heavy-tailed length map (5) determines the marginal of $h^l$, hence the
profile $|\phi'|$, hence the column-scale that feeds the structured
Wishart-Levy spectral theorem. The mean-field theory and the random-matrix
theory are therefore two pieces of a single picture, with the column
profile as their shared object.

## 5. The bivariate alpha-stable closure: $\mathcal{C}$-map analogue

For two inputs $x^1, x^2$, the joint distribution of $(h^l_i(x^1),
h^l_i(x^2))$ in the infinite-width limit follows from the same stability and
self-averaging steps as in Section 2, applied to the two-component vector
$\sum_j W^l_{ij}\,(\phi(h^{l-1}_j(x^1)), \phi(h^{l-1}_j(x^2)))$. The joint is
**bivariate symmetric $\alpha$-stable**, with characteristic function

$$
\hat p^l(t_1, t_2)
= \exp\!\Big( -\tfrac{1}{2}\, \sigma_w^\alpha\;
              \mathbb{E}_{(h^1, h^2) \sim P^{l-1}}
              \big| t_1\, \phi(h^1) + t_2\, \phi(h^2) \big|^\alpha \Big). \qquad (8)
$$

Here $P^{l-1}$ is the joint distribution of $(h^{l-1}(x^1), h^{l-1}(x^2))$ at
the previous layer, itself bivariate symmetric $\alpha$-stable by recursion.
Equation (8) is the **correlation-map analogue**: a recursion on the
*distribution* $P^l = \mathcal{C}_\alpha[P^{l-1}; \phi]$, not just on a
scalar correlation.

**Spectral-measure parameterisation.** Every bivariate symmetric
$\alpha$-stable distribution is parameterised by a spectral measure
$\Gamma$ on the unit circle $S^1$:

$$
\hat p(t_1, t_2)
= \exp\!\Big( -\tfrac{1}{2} \int_{S^1} |t_1 s_1 + t_2 s_2|^\alpha\,
              \Gamma(ds) \Big).
$$

Equation (8) is the statement that the layer-$l$ spectral measure is the
$\alpha$-th-moment angular distribution of the activation vector at the
previous layer, rescaled by $\sigma_w^\alpha$:

$$
\Gamma^l(A) = \sigma_w^\alpha\,
              \mathbb{E}\big[ R^\alpha\, \mathbf{1}[\Theta \in A] \big],
\qquad (R, \Theta) \text{ polar coords of } (\phi(h^1), \phi(h^2)) \sim P^{l-1},
\qquad (9)
$$

for any measurable $A \subseteq S^1$ (interpreting the symmetric stable so
that $\Gamma$ is symmetric under antipodal reflection).

For $\alpha = 2$ the spectral measure is non-unique (any measure with the
correct second moments suffices); the joint reduces to bivariate Gaussian
parameterised by a 2x2 covariance, and Poole's three scalar order parameters
$(q^l_{11}, q^l_{22}, q^l_{12})$ are recovered. For $\alpha \in (0, 2)$ the
spectral measure is essentially unique and is **genuinely
infinite-dimensional** -- no finite collection of scalars captures the joint
in general. The richer dependence structure of jointly $\alpha$-stable
random vectors is a structural difference from the Gaussian case, not just
a quantitative one.

**Marginal recovers the length map.** Setting $t_2 = 0$ in (8) gives
$\hat p^l(t, 0) = \exp(-\tfrac{1}{2}\, \sigma_w^\alpha\, \mathbb{E}|\phi(h^1)|^\alpha\,
|t|^\alpha)$, recovering the single-input length map (5).

**Empirical validation.** Test 5 of `heavy_tailed_mlp.ipynb` directly
checks eq. (8) by comparing the empirical bivariate CF
$\hat\varphi^l(t_1, t_2)$ computed across $N$ neurons at layer $l$
against the prediction $\exp(-\tfrac{1}{2}\sigma_w^\alpha\, (1/N)\sum_j
|t_1\phi(h^{l-1}_j(x^1)) + t_2\phi(h^{l-1}_j(x^2))|^\alpha)$ from the
previous-layer activations. At $\alpha = 1.5$, $\sigma_w = 1.0$,
$N \cdot \text{reps} = 6144$, $l = 5$, the empirical and predicted
$\log|\hat\varphi|$ agree to within max-abs $0.003$ across a fan of
$(t_1, t_2)$ directions and magnitudes -- well below the finite-$N$
noise floor.

**Operationally tractable regimes.** Two boundary cases are simple:
- **Fully decorrelated inputs.** $h^1, h^2$ independent; $\Gamma^l$
  concentrates on $\pm e_1, \pm e_2$ and the two marginals evolve
  independently via $V_\alpha$.
- **Fully correlated inputs.** $x^1 = x^2$, so $h^{l-1}(x^1) = h^{l-1}(x^2)$;
  $\Gamma^l$ concentrates on $\pm (1,1)/\sqrt 2$ and the joint is degenerate.

The interesting content for criticality is the **stability of the
fully-correlated fixed point** -- how a small input perturbation
$x^2 = x^1 + \delta x$ propagates through depth.

## 6. Perturbation analysis: bivariate-stable recursion

Linearise around the fully-correlated solution. Define the layer-$l$
perturbation $\xi^l_i := h^l_i(x^2) - h^l_i(x^1)$ and its $\alpha$-scale
$T^l$ intrinsically as the stability scale of the (asymptotically
$\alpha$-stable) marginal of $\xi^l$, via
$\hat p^l_\xi(t) = \exp(-\tfrac{1}{2} (T^l)^\alpha |t|^\alpha)$. Assume the
perturbation is small relative to the pre-activation, $T^{l-1} \ll S^{l-1}$.

### Exact unconditional recursion -- eq. (10)

Setting $(t_1, t_2) = (-t, t)$ in the bivariate-stable CF (8) extracts the
marginal CF of $\xi^l$:

$$
\hat p^l_\xi(t)
= \hat p^l(-t, t)
= \exp\!\Big( -\tfrac{1}{2}\, \sigma_w^\alpha\, |t|^\alpha\,
              \mathbb{E}\big[ |\phi(h^2) - \phi(h^1)|^\alpha \big] \Big),
$$

so the perturbation $\alpha$-scale obeys the **exact unconditional recursion**

$$
\boxed{\quad
(T^l)^\alpha
\;=\; \sigma_w^\alpha\,
      \mathbb{E}_{(h^1, h^2) \sim P^{l-1}}
      \big[ |\phi(h^2) - \phi(h^1)|^\alpha \big],
\quad} \qquad (10)
$$

where $P^{l-1}$ is the bivariate symmetric $\alpha$-stable joint of
$(h^{l-1}(x^1), h^{l-1}(x^2))$ at the previous layer. This is finite for
bounded $\phi$ and depends on the *full joint* $P^{l-1}$, not on any scalar
correlation.

### Linearisation in the small-perturbation regime -- eq. (11)

For small $\xi^{l-1}$, $\phi(h^{l-1}(x^2)) - \phi(h^{l-1}(x^1)) =
\phi'(h^{l-1}(x^1))\, \xi^{l-1} + O((\xi^{l-1})^2)$, so

$$
(T^l)^\alpha = \sigma_w^\alpha\,
               \mathbb{E}\big[ |\phi'(h^{l-1}(x^1))\, \xi^{l-1}|^\alpha \big]
             + O\big((T^{l-1})^{2\alpha} \big), \qquad (11)
$$

where the expectation is over the *joint* distribution of
$(\phi'(h^{l-1}), \xi^{l-1})$ at a single neuron. In the infinite-width
limit this joint is itself bivariate symmetric $\alpha$-stable (by the same
argument as the C-map, applied to $(h^{l-1}, \xi^{l-1})$).

### Closure step fails empirically: operational EoC via Lyapunov exponent

The natural Poole-style next move is to close (11) scalarly by assuming
the joint $(\phi'(h^{l-1}), \xi^{l-1})$ factorises in the infinite-width
limit -- the strong-form mean-field decoupling. That would give

$$
(T^l)^\alpha = \chi_1|_\alpha^{\text{naive}}\, (T^{l-1})^\alpha,
\qquad
\chi_1|_\alpha^{\text{naive}}
:= \sigma_w^\alpha \int p_\alpha(z)\,
   \big|\phi'\big((q^*)^{1/\alpha} z\big)\big|^\alpha\, dz,
$$

reducing to Poole's $\chi_1$ at $\alpha = 2$ (where decoupling is rigorous
by joint-Gaussianity $+$ zero covariance $\Rightarrow$ independence).

**For $\alpha < 2$ the decoupling assumption is empirically false.**
Direct simulation (`heavy_tailed_mlp.ipynb`, Test 1) shows that the
moment-factorisation ratio

$$
R := \frac{\mathbb{E}|\phi'(h)\,\xi|^\alpha}
          {\mathbb{E}|\phi'(h)|^\alpha\;\mathbb{E}|\xi|^\alpha}
$$

does *not* approach $1$ as the width grows: at $\alpha = 1.5$,
$\sigma_w = 1.0$, tanh, $R$ plateaus at $\approx 0.55$--$0.65$ across
$N \in \{64, \dots, 4096\}$. Test 3 corroborates the failure at the
joint level: the $R^\alpha$-weighted angular density of $(h, \xi)$ has
about a third of its mass off the coordinate axes
($\sim 0.68$ near-axis fraction vs $\sim 0.50$ for uniform and $\to 1$
for strong decoupling). The Gaussian decoupling-by-symmetry argument
does not survive $\alpha < 2$: vanishing covariation does not imply
independence for $\alpha$-stable variables, and the symmetry
$\xi \leftrightarrow -\xi$ forces the joint spectral measure to be
reflection-invariant but does not force it onto the coordinate axes.

The consequences are quantitative as well as structural. At
$\alpha = 1.5$, $\sigma_w = 1.0$: $\chi_1|_\alpha^{\text{naive}} = 0.91 < 1$
predicts decay, but empirical $\hat T^l$ (Test 2) *grows* at $\sim 1.15
\times$ per layer; the naive closure gets even the sign of the dynamics
wrong.

**Operational edge of chaos: the Lyapunov exponent $\lambda$ of
$\mathcal{F}$.** Because (11) does not close scalarly, the operational
EoC is read off directly from the nonlinear forward dynamics:

$$
\lambda(\alpha, \sigma_w)
\;:=\; \lim_{l \to \infty}\, \tfrac{1}{l}\, \log\, \hat T^l / T^0,
$$

measured by iterating the forward MLP at infinite width and fitting the
per-layer growth rate of $T^l$. EoC is $\lambda(\alpha, \sigma_w) = 0$.

Test 4 of `heavy_tailed_mlp.ipynb` measures $\lambda$ on a 10x11 grid
$(\alpha, \sigma_w) \in [1.1, 2.0] \times [0.5, 2.5]$ ($N = 1024$,
$L = 25$, 3 reps per cell). The empirical critical $\sigma_w$ sits
*left of* the naive $\chi_1|_\alpha^{\text{naive}} = 1$ curve across the
heavy-tailed regime, with the gap shrinking monotonically as $\alpha \to 2$:

| $\alpha$ | empirical $\sigma_w^c$ ($\lambda = 0$) | naive $\sigma_w^c$ ($\chi_1|_\alpha^{\text{naive}} = 1$) | shift |
|---|---|---|---|
| 1.30 | 0.73 | 1.14 | $-0.41$ |
| 1.70 | 0.92 | 1.12 | $-0.20$ |
| 1.80 | 0.93 | 1.11 | $-0.18$ |
| 1.90 | 1.01 | 1.10 | $-0.09$ |

(Stripes at $\alpha \in \{1.1, 1.2, 1.4, 1.5, 1.6\}$: empirical $\sigma_w^c$
lies below 0.5 and is off-grid. Stripe at $\alpha = 2.0$: both quantities
are below 1 throughout the visible range, consistent with the Gaussian
threshold at $\sigma_w = 1.0$ sitting close to where $q^*$ becomes
nontrivial.) The heavy-tailed network reaches criticality at *smaller*
$\sigma_w$ than the naive scalar would predict, and the deviation grows
with how heavy-tailed the weights are. The natural correlation depth
scale is $1/|\lambda|$; it does not have a closed form in
$\chi_1|_\alpha^{\text{naive}}$.

## 7. Reduction to Poole's $\mathcal{C}$-map and $\chi_1$ at $\alpha = 2$

At $\alpha = 2$, $p_2(z)\, dz = Dz$ and the naive scalar slope of Section 6 reduces to

$$
\chi_1|_\alpha^{\text{naive}}\Big|_{\alpha=2}
= \sigma_w^2 \int Dz\, \phi'(\sqrt{q^*}\, z)^2
\;\equiv\; \chi_1,
$$

which is Poole et al. (2016)'s $\chi_1$. **At $\alpha = 2$ the naive slope
is the correct slope**: the mean-field decoupling holds by the
Gaussian-covariance argument, so the linearised joint-evolution operator
on the bivariate Gaussian joint reduces to scalar multiplication by
$\chi_1$. The "naive slope is wrong" caveat of Section 6.3 applies only to
$\alpha < 2$. The bivariate-stable joint (8) collapses to bivariate
Gaussian:

$$
\sigma_w^2\, \mathbb{E}\big[ t_1\,\phi(h^1) + t_2\,\phi(h^2) \big]^2
= \sigma_w^2 \big[
    t_1^2\,\mathbb{E}\phi(h^1)^2
    + 2 t_1 t_2\, \mathbb{E}\phi(h^1)\phi(h^2)
    + t_2^2\,\mathbb{E}\phi(h^2)^2
  \big],
$$

which is Poole's $(q^l_{11}, q^l_{22}, q^l_{12})$ recursion. After
normalisation $c^l_{12} = q^l_{12} / \sqrt{q^l_{11} q^l_{22}}$ this is
Poole's $\mathcal{C}$-map, with slope at $c = 1$ equal to
$\chi_1|_\alpha^{\text{naive}}|_{\alpha=2} = \chi_1$. The heavy-tailed
framework reduces to Poole's framework on the nose at $\alpha = 2$.

*Notation.* We write $\chi_1|_\alpha^{\text{naive}}$ rather than
$\chi_1|_\alpha$ throughout the heavy-tailed sections to flag that for
$\alpha < 2$ it is the would-be slope under decoupling, not an operational
quantity. The bar-without-superscript form is reserved for the $\alpha = 2$
case where it agrees with Poole's $\chi_1$ on the nose.

## 8. The joint state space: spectral measures on $S^1$

A structural appendix to Section 6: this section names the natural
state space for the heavy-tailed dynamics (spectral measures on $S^1$),
verifies the fully-correlated fixed point used implicitly in the
perturbation analysis, and records an auxiliary diagonal-entry
quantity from the exact bivariate-stable closure (16) that doubles as
a sanity check on Sections 5--6 and on the $\alpha = 2$ reduction.

### 8.1 The bivariate-stable joint as state space

The state at layer $l$ is a positive measure $\Gamma^l$ on $S^1$, invariant
under the antipodal map $s \mapsto -s$. The joint of $(h^l(x^1), h^l(x^2))$
is bivariate symmetric $\alpha$-stable with characteristic exponent

$$
\psi_\Gamma(t) \;=\; \int_{S^1} |\langle t, s\rangle|^\alpha\, \Gamma(ds),
\qquad t \in \mathbb{R}^2. \qquad (14)
$$

Marginal scales are recovered as
$\langle |s_1|^\alpha, \Gamma\rangle = q^l_{11}$,
$\langle |s_2|^\alpha, \Gamma\rangle = q^l_{22}$,
both equal to $q^*$ at the fully-correlated fixed point.

*Convention.* Throughout we adopt the raw-pairing convention $\langle f, \mu\rangle :=
\int_{S^1} f(s)\,\mu(ds)$ with no symmetrisation prefactor, so e.g.
$\langle s_1 s_2, \Gamma\rangle = q_{12}$ at $\alpha = 2$ (the standard
covariance entry, not half of it); this fixes the factor of $2$ in the
binomial expansion of $(t_1 s_1 + t_2 s_2)^2$ that propagates to the
Poole reduction in Section 8.5.

For $\alpha = 2$ the spectral measure is non-unique: bivariate Gaussian is
determined by the $2\times 2$ covariance $\Sigma = (q_{11}, q_{12}; q_{12},
q_{22})$, and many $\Gamma$ realise the same $\Sigma$. The dependence on
$\Gamma$ at $\alpha = 2$ collapses to dependence on $\Sigma$. For $\alpha < 2$
the spectral measure is essentially unique and *infinite-dimensional* as
an object on $S^1$; no finite collection of moments captures the joint.
This is the technical content of "bivariate stability has richer
dependence structure than bivariate Gaussian" used throughout Sections
5--7.

### 8.2 Action of $\mathcal{F}$

For any continuous test function $g: S^1 \to \mathbb{R}$, $\mathcal{F}[\Gamma]$
acts as
$$
\langle g, \mathcal{F}[\Gamma]\rangle
\;=\; \sigma_w^\alpha\, \mathbb{E}_{(h^1, h^2) \sim P_\Gamma}\!\left[
   \big(\phi(h^1)^2 + \phi(h^2)^2\big)^{\alpha/2}\, g(\Theta)
\right], \qquad (16)
$$

where $P_\Gamma$ is the bivariate $\alpha$-stable joint with spectral measure
$\Gamma$ (CF eq. 14), and $\Theta = (\phi(h^1), \phi(h^2))/R$ is the polar
angle of $(\phi(h^1), \phi(h^2))$ on $S^1$.

### 8.3 Fully-correlated fixed point $\Gamma^*$

$\Gamma^*$ is supported on $\pm e_d/\sqrt{2}$ where $e_d = (1,1)$, with total
mass $m^* = 2^{\alpha/2} q^*$. The marginal $\alpha$-scale of $h^1$ at
$\Gamma^*$ is then $\langle |s_1|^\alpha, \Gamma^*\rangle = m^* \cdot
(1/\sqrt 2)^\alpha = q^*$, as required.

Verification $\mathcal{F}[\Gamma^*] = \Gamma^*$: at $\Gamma^*$, $(h^1, h^2)
\stackrel{d}{=} (h, h)$ with $h$ symmetric $\alpha$-stable of scale
$(q^*)^{1/\alpha}$ (the diagonal collapse). Then $R = \sqrt{2}|\phi(h)|$ and
$\Theta = \mathrm{sign}(\phi(h))\, e_d/\sqrt{2}$. For odd $\phi$ and
symmetric $h$, $\Theta$ is uniform on $\pm e_d/\sqrt{2}$. Substituting into
(16),
$$
\langle g, \mathcal{F}[\Gamma^*]\rangle
= \sigma_w^\alpha\, 2^{\alpha/2}\, \mathbb{E}|\phi(h)|^\alpha \cdot \tfrac{1}{2}
  \big(g(e_d/\sqrt 2) + g(-e_d/\sqrt 2)\big),
$$
which equals $\langle g, \Gamma^*\rangle$ once $\sigma_w^\alpha \mathbb{E}|\phi(h)|^\alpha
= V_\alpha(q^*) = q^*$. The length-map fixed point is exactly the
self-consistency for $\mathcal{F}$ on the fully-correlated configuration.

### 8.4 An auxiliary observation: the operator-level diagonal entry

Beyond the naive scalar slope $\chi_1|_\alpha^{\text{naive}}$ of Section 6
and the operational Lyapunov exponent $\lambda$, a third quantity arises
naturally from (16) as a structural by-product:

$$
L_{q_{12}, q_{12}} \;:=\;
   \frac{d}{d\epsilon}\bigg|_{\epsilon = 0}\,
   \langle s_1 s_2,\, \mathcal{F}[\Gamma^* + \epsilon\,\eta_{q_{12}}]\rangle
$$

where $\eta_{q_{12}}$ is the perturbation with
$\langle s_1 s_2, \eta_{q_{12}}\rangle = 1$ and
$\langle |s_j|^\alpha, \eta_{q_{12}}\rangle = 0$ for $j = 1, 2$ -- the
sensitivity of the next-layer $q_{12}$ to a current-layer $q_{12}$
perturbation, computed directly from the exact functional (16) without
the decoupling assumption that produces $\chi_1|_\alpha^{\text{naive}}$.

At $\alpha = 2$ the three quantities $\chi_1|_\alpha^{\text{naive}}$,
$L_{q_{12}, q_{12}}$, and the per-step Lyapunov factor $e^\lambda$ all
coincide (with common value Poole's $\chi_1$; see Section 8.5). For
$\alpha < 2$ they are distinct, and the distinctions probe different
failure modes: $\chi_1|_\alpha^{\text{naive}}$ vs $L_{q_{12}, q_{12}}$
tests whether the decoupling assumption holds at the fixed point
$\Gamma^*$ (a *static* property of the joint), while
$L_{q_{12}, q_{12}}$ vs $e^\lambda$ tests whether the linearised
$q_{12}$-direction captures the full perturbation dynamics (a *dynamical*
property of the forward iteration $\mathcal{F}$).

Empirically (probe `.agents/temp/L_q12_probe.py`, Monte Carlo with
$n = 2000$, $r_{\max} = 30$, $n_r = 400$, at $\sigma_w = 1.5$):

| $\alpha$ | $\chi_1|_\alpha^{\text{naive}}$ | $L_{q_{12}, q_{12}}$ | naive $- L_{q_{12}, q_{12}}$ |
|---|---|---|---|
| 1.50 | 1.165 | $0.901 \pm 0.010$ | $+0.264$ |
| 1.70 | 1.156 | $1.016 \pm 0.012$ | $+0.140$ |
| 1.90 | 1.144 | $1.101 \pm 0.013$ | $+0.043$ |
| 1.95 | 1.142 | $1.115 \pm 0.013$ | $+0.027$ |
| 1.99 | 1.140 | $1.124 \pm 0.014$ | $+0.016$ |
| 2.00 | 1.139 | $1.132 \pm 0.014$ | $+0.007$ |

The operator-level diagonal is systematically *smaller* than the naive
integral for $\alpha < 2$, with the gap monotonically vanishing as
$\alpha \to 2$ -- a quantitative measure of how the joint structure at
$\Gamma^*$ correlates $\phi'(h)$ and $\xi$ in a way that, at the moment
level, reduces the effective slope below what factorisation predicts.
This quantity has no operational role in the heavy-tailed EoC criterion
(which is $\lambda = 0$, measured by Test 4), but it is a real
structural fact about the joint dynamics and a useful sanity check on
the bivariate-stable closure (8): the $\alpha = 2$ row of the table
recovers Poole's $\chi_1$ exactly, validating (8) and (16) numerically.

### 8.5 Reduction to Poole at $\alpha = 2$

At $\alpha = 2$, the spectral measure parameterisation degenerates:
bivariate Gaussian is fully specified by the covariance $\Sigma$, with
many $\Gamma$ realising the same $\Sigma$. The joint perturbation space
collapses to the single direction $\delta q_{12}$, and the three
quantities of Sections 6 and 8.4 all coincide with Poole's $\chi_1$.

**Explicit reduction.** Take $\eta_{q_{12}}$ as in Section 8.4. The
characteristic exponent perturbs as
$\psi_{\eta_{q_{12}}}(t) = \int (t_1 s_1 + t_2 s_2)^2\, \eta_{q_{12}}(ds)
= 2 t_1 t_2$. The joint CF perturbs multiplicatively, $\delta \hat p(t)
= -\tfrac{\epsilon}{2}\cdot 2 t_1 t_2\, \hat p_{\Gamma^*}(t)$, which by
Fourier inversion gives $\delta p(h) = \epsilon\, \partial^2_{h^1 h^2}
p_{\Gamma^*}(h)$. For any smooth $\Phi$, two integrations by parts give
$\delta \mathbb{E}[\Phi] = \epsilon\, \mathbb{E}_{\Gamma^*}[\partial^2_{h^1 h^2}
\Phi]$. Specialising $\Phi(h) = \phi(h^1)\phi(h^2)$ (which is the
$q_{12}$ component of $\mathcal{F}[\Gamma]$ at $\alpha = 2$ via (16)),
$\partial^2_{h^1 h^2}\Phi = \phi'(h^1)\phi'(h^2)$. At the fully-correlated
fixed point $h^1 = h^2 = h$ this is $\phi'(h)^2$, so

$$
L_{q_{12}, q_{12}}\Big|_{\alpha = 2}
\;=\; \sigma_w^2\, \mathbb{E}[\phi'(h)^2]
\;=\; \chi_1.
$$

The $\alpha = 1.9$ row of the table above ($L_{q_{12}, q_{12}} = 1.101$,
naive $= 1.144$, gap $0.043$) trending to $0$ at $\alpha = 2$ is the
empirical witness of this collapse, paralleled by Test 4's $\alpha = 1.9$
EoC shift of $-0.09$ trending to $0$ at $\alpha = 2$.

---

## Next derivation

Two independent extensions of the heavy-tailed MFT, neither depending on a separate operator-level numerical follow-on:

- **Backward / gradient analysis.** Heavy-tailed analogue of Schoenholz 2017. The naive $\chi_1|_\alpha^{\text{naive}}$ no more controls backward gradient magnitude than it controls forward perturbation growth (Sections 6.3 and 8.4 establish that decoupling fails for $\alpha < 2$). The right object is the bivariate-stable joint of $(\delta^l, \delta^l)$ across the two inputs; the recursion is the dual of $\mathcal{F}$ acting on this joint. Open.

- **Connection to `RMT/structured_wishart_levy.md`.** *Realised in
  `ht_mlp_jacobian.md`* (derivation), `ht_mlp_jacobian.py` (validation),
  `ht_mlp_jacobian.ipynb` (visualisation). The column profile
  $c(y) = |\phi^\prime(h^{l-1}(y))|$ at the heavy-tailed $q^*$ feeds the
  structured Wishart-Levy Theorem 2 via a random-to-deterministic quantile
  embedding (Lemma 1 of `ht_mlp_jacobian.md`), giving the layerwise
  Jacobian SV density as a single scalar fixed point in $Y_r$ at
  $\gamma = 1$ -- agreement with population dynamics, synthetic and
  MLP-derived empirical SVDs is verified at bin-noise tolerance.

- **Validation.** Cross-check follow-on predictions against `heavy_tailed_mlp.ipynb` -- Test 4 empirical edge of chaos is the standing benchmark for any heavy-tailed criticality statement.
