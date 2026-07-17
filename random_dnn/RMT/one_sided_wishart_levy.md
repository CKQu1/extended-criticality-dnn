# One-sided heavy-tailed Wishart-Levy law (Theorem 2)

Numerical solver dedicated to Theorem 2 of `structured_wishart_levy.md`:
the case $|\tau(x, y)| = c(y)$ depending on the column coordinate only.
Math derivation, proofs, and the $\tau$-general Theorem 1 framework are in
`structured_wishart_levy.md`; this note documents only what is specific to
the one-sided scalar reduction and the resulting numerical scheme.

References to `wishart-amir`, `weakening`, `gammaone`, `rhoalpha` and to
the kernels $g_\alpha$, $h_\alpha$, $C_\alpha$, $\mathcal{K}_\alpha$ are
as in `structured_wishart_levy.md` Preliminaries.

## 1. The Theorem 2 scalar closure

Recap (`structured_wishart_levy.md`, Theorem 2). Under one-sided
$|\tau(x, y)| = c(y)$, the field $Y_r(x, z)$ of Theorem 1 collapses to a
single scalar $Y_r(z) : \mathbb{C}^+ \to \mathcal{K}_\alpha$, solving the
**nested scalar closure**

$$
z^\alpha Y_r
\;=\; \frac{\gamma}{1 + \gamma}\, C_\alpha
\int_0^1 c(v)^\alpha\,
g_\alpha\!\Big(
\frac{C_\alpha\, c(v)^\alpha}{(1 + \gamma)\, z^\alpha}\,
g_\alpha(Y_r)
\Big)\, dv. \qquad (1)
$$

The squared-singular-value law $\nu$ has atom $1 - \gamma$ at $0$ and a
continuous density on $(0, \infty)$ given by the **scalar collapse**

$$
G_\nu(\zeta)
\;=\; \frac{1}{\zeta}\, h_\alpha\!\big(Y_r(\sqrt\zeta)\big),
\qquad
\rho_\nu(t)
\;=\; -\frac{1}{\pi t}\, \Im\, h_\alpha\!\big(Y_r(\sqrt t)\big). \qquad (2)
$$

In Gram convention the singular-value density is $f_{\mathrm{SV}}(s) =
2 s\, \rho_\nu(s^2)$.

**Tail** (specialisation of Theorem 1(v) to one-sided $\tau$): as
$t \to \infty$,

$$
t^{1 + \alpha/2}\, \rho_\nu(t)
\;\longrightarrow\;
\frac{\alpha\, \gamma}{2(1 + \gamma)}\, \int_0^1 c(v)^\alpha\, dv,
\qquad
f_{\mathrm{SV}}(s) \sim B\, s^{-1-\alpha},
\quad B = \frac{\alpha\, \gamma}{1 + \gamma}\, \int_0^1 c^\alpha. \qquad (3)
$$

## 2. The $h_\alpha$-rule choice (why a dedicated solver helps)

`structured_wishart_levy.py` evaluates $h_\alpha$ from the *shared*
Gauss-Laguerre rule used for $g_\alpha$ (via the identity
$h_\alpha = 1 - \tfrac{\alpha}{2}\, y\, g_\alpha(y)$), because the
Theorem 1 collapse over the vector field $Y_r(x)$ requires the discrete
identity $\Delta_r \langle Y_r\, g_\alpha(Y_r)\rangle = \Delta_c
\langle Y_c\, g_\alpha(Y_c)\rangle$ to hold exactly under quadrature
(`structured_wishart_levy.md`, Step 4 of Theorem 1's proof + Quadrature
consistency remark). Numerically the shared-rule $h_\alpha$ has
$\sim 5 \times 10^{-3}$ absolute error in $\Im h_\alpha$; via the density
readout $\rho \propto \Im h / s$ this gets amplified at small $s$.

The Theorem 2 scalar collapse (2) **does not require this discrete
identity** -- the readout is a single $h_\alpha$ evaluation at a scalar
$Y_r$, with no sum-rule to preserve. The independent-rule $h_\alpha$
(its own Gauss-Laguerre integral
$h_\alpha(y) = \int_0^\infty e^{-t} e^{-t^{\alpha/2} y}\, dt$, as used in
`wishart_levy.py`) gives a more accurate $\Im h_\alpha$
at the cost of
breaking the structured collapse identity -- which is irrelevant here.

This module therefore solves the field (1) with the shared-rule
$g_\alpha$ engine of `structured_wishart_levy._scalar_one_sided_closure`
(so the field $Y_r$ is identical to the structured solver's value to
machine precision -- see Gate 2 below) but reads off the density via the
independent-rule $h_\alpha$ of `wishart_levy.h_alpha`. The split is the
operational content of "Theorem 2 doesn't need the structured collapse;
use independent-rule $h_\alpha$ where it matters."

## 3. Numerical scheme

- *Quadrature.* Gauss-Legendre on $[0, 1]$ for the $c$-integral
  (`profile_order` nodes, weights sum to $1$), Gauss-Laguerre for both
  $g_\alpha$ (in the field) and the independent $h_\alpha$ (in the
  readout) (`quadrature_order` nodes each).
- *Sweep direction.* Solve $z$-grid from $|z|$-large to $|z|$-small,
  threading the previous solution as the Newton seed (the
  $|z| \to \infty$ asymptote $Y_r \sim C_\alpha g_\alpha(0) / [(1 +
  \gamma) z^\alpha]$ anchors the high-$|z|$ end). Same continuation
  strategy as `structured_wishart_levy.py` and
  `wishart_levy.py`.
- *Seed priority.* `wishart_levy.solve_y_pair` (Corollary 1 anchor at
  this $z$), then the carried previous-step solution, then the
  asymptote. Newton-Raphson via `scipy.optimize.root(method='hybr')` on
  the real/imaginary pair (real-valued residual of size $2$).
- *Normalisation.* Same SciPy-vs-Belinschi convention as the rest of
  `RMT/`: `entry_scale` is the SciPy stable scale of the underlying
  $\alpha$-stable variables; `normalization='stable'` returns
  singular values in physical (SciPy-entry-scale) units while
  `'belinschi'` returns BDG-canonical units. The conversion factor is
  `wishart_levy._output_scale`.

## 4. Validation gates

- **Gate A: constant $c \equiv $ const (Corollary 1 of
  `structured_wishart_levy.md`).** Reduces to `wishart_levy`. Field
  $Y_r$ should agree to $10^{-9}$ (same scalar closure with the same
  numerical $g_\alpha$ rule); density should agree to $10^{-9}$ (both
  sides use independent-rule $h_\alpha$, so no quadrature mismatch).
  Implemented as `compare_constant_to_wishart`.

- **Gate B: general one-sided $c(v)$ -- field-level vs structured
  solver.** The field $Y_r$ should agree with the
  `_scalar_one_sided_closure` of `structured_wishart_levy.py` to
  $10^{-9}$ (same equation, same $g_\alpha$ rule). Implemented as
  `compare_field_to_structured`. *Density* values disagree at small
  $s$ in proportion to the shared-vs-independent $h_\alpha$ error --
  this is the documented improvement; the gate prints the gap.

- **Gate C: heavy-tailed MLP Jacobian** (downstream).
  `ht_mlp_jacobian.py:run_validation` cross-checks the one-sided curve
  against (P2) population-dynamics cavity equations and (P3a / P3b)
  empirical SVDs. With the independent-rule $h_\alpha$ readout, the
  small-$s$ Gate-A-style accuracy carries through to the
  non-constant-$c$ case as well.

## 5. Scope

- One-sided ($|\tau(x, y)| = c(y)$ only). For genuinely two-sided
  $\tau$, the row field $Y_r(x)$ stays functional and only
  `structured_wishart_levy.py` (Theorem 1) applies -- there is no
  scalar reduction to a 1-D solver.
- The atom $1 - \gamma$ at $0$ is a separate component, returned via
  the curve dataclass. The continuous density is for $(0, \infty)$.
- $\alpha = 2$ is the Gaussian limit; `belinschi_quantile_scale` is
  degenerate there (`structured_wishart_levy.md` Specialisations
  remark). For practical use restrict $\alpha \in (0, 2)$; at
  $\alpha = 2$ use a Marchenko-Pastur-with-profile path separately.
