# The structured Wishart-Levy law

Two theorems (with proofs). **Theorem 1** generalizes Theorem `wishart-amir`
of Belinschi-Dembo-Guionnet to a rectangular heavy-tailed matrix $X_{N,M}$
carrying its **own deterministic entry profile**; it is written so as to
**replace Theorem `wishart-amir`** in the paper (a constant profile recovers it
verbatim). **Theorem 2** specializes to a **one-sided profile** (the
column-scaling case), where the row field collapses to a single scalar and the
singular-value density is expressed through that scalar fixed-point field; the
degenerate Wigner/plain-square corner additionally inherits the algebraic
density and the qualitative features of `wishart-amir`.

References to `weakening`, `wishart-amir`, `gammaone`, `eq:BAG6`, `rhoalpha`
are to `.agents/notes/belinschi-2008-preprint/wishart-arxiv-rev.tex`; sibling
notes are `belinschi-2008.md` and `hermitisation.md`. The structured law
(both theorems) is implemented in `structured_wishart_levy.py`.

## Preliminaries

Let $(x_{ij})$ be i.i.d. real with
$\Pr(|x_{ij}|\ge u)=L(u)u^{-\alpha}$, $\alpha\in(0,2)$, $L$ slowly varying, and
$a_N=\inf\{u:\Pr(|x_{ij}|\ge u)\le 1/N\}$ the BDG quantile normalization.
Define, as in the paper,
$$
g_\alpha(y)=\int_0^\infty t^{\frac\alpha2-1}e^{-t}e^{-t^{\alpha/2}y}\,dt,
\qquad
h_\alpha(y)=\int_0^\infty e^{-t}e^{-t^{\alpha/2}y}\,dt
=1-\tfrac\alpha2\,y\,g_\alpha(y),
$$
$$
C_\alpha=i^\alpha\,\frac{\Gamma(1-\tfrac\alpha2)}{\Gamma(\tfrac\alpha2)},
\quad (i)^\alpha=e^{i\pi\alpha/2},
\qquad
\mathcal K_\alpha=\{Re^{i\theta}:|\theta|\le\tfrac{\alpha\pi}2,\,R\ge0\}.
$$

**Structured matrix.** Let $\tau:[0,1]_{\mathrm{row}}\times[0,1]_{\mathrm{col}}
\to\mathbb R$ be **deterministic** and put
$$
X_{N,M}(i,j)=\tau\!\left(\tfrac iN,\tfrac jM\right)x_{ij},
\qquad X_{N,M}\in\mathbb R^{N\times M},\quad \frac MN\to\gamma\in(0,1],
$$
$$
W_{N,M}=a_{N+M}^{-2}\,X_{N,M}X_{N,M}^{\mathsf t}.
$$

**Embedding.** With $\Delta_r=\tfrac1{1+\gamma}$,
$\Delta_c=\tfrac\gamma{1+\gamma}$, $I_r=(0,\Delta_r]$, $I_c=(\Delta_r,1]$, and
the affine charts $\hat x=(1+\gamma)x$ on $I_r$,
$\hat y=\tfrac{1+\gamma}\gamma(y-\Delta_r)$ on $I_c$, define the symmetric
profile on $[0,1]^2$
$$
\sigma_\tau(x,y)=
\begin{cases}
\tau(\hat x,\hat y), & x\in I_r,\ y\in I_c,\\
\tau(\hat y,\hat x), & x\in I_c,\ y\in I_r,\\
0, & \text{otherwise.}
\end{cases}
$$
This is the $q=2$, $\sigma_{rs}=\mathbf 1_{|r-s|=1}$ pattern of `wishart-amir`
with the off-diagonal block profile-weighted.

**Admissibility.** Say $\tau\in\mathcal F_\alpha^{\mathrm{rect}}$ if
$\big\|\int_0^1|\tau(\cdot,v)|^\alpha dv\big\|_\infty<\infty$ and
$\big\|\int_0^1|\tau(w,\cdot)|^\alpha dw\big\|_\infty<\infty$, with
$|\tau|^\alpha$ piecewise-constant-approximable on each axis (equivalently
$\sigma_\tau\in\mathcal F_\alpha$ in the sense of the paper). Bounded $\tau$
with finitely many jumps qualifies.

For $z\in\mathbb C^+$ write $\langle f\rangle:=\int_0^1 f(x)\,dx$.

**Squared vs. singular values.** $\nu=\mu_\alpha^{\gamma,\tau}$, the limit of
$\hat\mu_{W_{N,M}}$, is the law of the **squared** singular values of
$a_{N+M}^{-1}X_{N,M}$: a probability measure with an atom of mass $1-\gamma$ at
$0$ and continuous mass $\gamma$ on $(0,\infty)$ (Theorems 1-2 are about
$\nu$, with continuous density $\rho_\nu$). To compare with singular values
$s=\sqrt t$ there are two conventions, differing by the factor $\gamma$: the
**atomless** law of the $M$ actual singular values, $f_{\mathrm{SV}}(s)
=\tfrac{2s}{\gamma}\rho_\nu(s^2)$ (mass $1$ -- an empirical SVD histogram), and
the **Gram** law $f^{(N)}_{\mathrm{SV}}(s)=2s\,\rho_\nu(s^2)$ (mass $\gamma$,
atom $1-\gamma$ -- the convention used by `structured_wishart_levy.py`).

---

## Theorem 1 (Structured Wishart-Levy law)

*Let $\tau\in\mathcal F_\alpha^{\mathrm{rect}}$ be deterministic and
$M/N\to\gamma\in(0,1]$. Then:*

**(i)** *$\hat\mu_{W_{N,M}}$ converges weakly to a non-random probability
measure $\mu_\alpha^{\gamma,\tau}$ -- almost surely if $\tau$ is
piecewise-constant-equivalent on both axes, otherwise in expectation --
with an atom at $0$ of mass $1-\gamma$.*

**(ii)** *There is a unique pair of analytic mappings
$Y_r,Y_c:\mathbb C^+\to L^\infty([0,1];\mathcal K_\alpha)$, each tending to $0$
as $|z|\to\infty$, such that for $|z|\ge R(\tau)$ and a.e. $x,y$,*
$$
z^\alpha Y_r(x,z)=\frac{\gamma}{1+\gamma}\,C_\alpha
\int_0^1|\tau(x,v)|^\alpha g_\alpha\!\big(Y_c(v,z)\big)\,dv,
$$
$$
z^\alpha Y_c(y,z)=\frac{1}{1+\gamma}\,C_\alpha
\int_0^1|\tau(w,y)|^\alpha g_\alpha\!\big(Y_r(w,z)\big)\,dw.
$$

**(iii) (Collapse).** *The Cauchy-Stieltjes transform of
$\mu_\alpha^{\gamma,\tau}$ is, for $\zeta\in\mathbb C^+$,*
$$
G_\nu(\zeta)=\int\frac{d\mu_\alpha^{\gamma,\tau}(t)}{\zeta-t}
=\frac1\zeta\int_0^1 h_\alpha\!\big(Y_r(x,\sqrt\zeta)\big)\,dx .
$$

**(iv) (Density).** *On $(0,\infty)$, $\mu_\alpha^{\gamma,\tau}$ has the
continuous density, bounded off zero,*
$$
\rho_\alpha^{\gamma,\tau}(t)
=-\frac1{\pi t}\,\Im\int_0^1 h_\alpha\!\big(Y_r(x,\sqrt t)\big)\,dx .
$$

**(v) (Tail).** *As $t\to\infty$,*
$$
t^{1+\alpha/2}\,\rho_\alpha^{\gamma,\tau}(t)
\;\longrightarrow\;
\frac{\alpha\gamma}{2(1+\gamma)}
\iint_{[0,1]^2}|\tau(x,y)|^\alpha\,dx\,dy .
$$

**Corollary 1 (Reduction to `wishart-amir`).** *If $\tau\equiv1$ then
$Y_r,Y_c$ are constant in $x,y$, equal to the functions $Y_1,Y_2$ of
`wishart-amir`; (ii) becomes the two-function system
$z^\alpha Y_1=\tfrac{\gamma}{1+\gamma}C_\alpha g_\alpha(Y_2)$,
$z^\alpha Y_2=\tfrac1{1+\gamma}C_\alpha g_\alpha(Y_1)$; (iii)-(v) become
$G_\nu=\tfrac1\zeta h_\alpha(Y_1(\sqrt\zeta))$, the atom $1-\gamma$, the density
$-\tfrac1{\pi t}\Im h_\alpha(Y_1(\sqrt t))$, and the tail constant
$\tfrac{\alpha\gamma}{2(1+\gamma)}$ -- i.e. Theorem `wishart-amir` verbatim. For
$\gamma=1$ this further reduces, by the paper's identity, to
$\rho^1_\alpha(t)=2^{1/\alpha}t^{-1/2}\rho_\alpha(2^{1/\alpha}\sqrt t)$ with
$\rho_\alpha$ the Wigner-Levy density.*

### Proof of Theorem 1

**Step 1 (embedding is admissible).** $\sigma_\tau$ is symmetric by
construction. Its two marginal operator norms are exactly the two finite
quantities in the definition of $\mathcal F_\alpha^{\mathrm{rect}}$ (the
$x\in I_r$ slice integrates $|\tau(\hat x,\cdot)|^\alpha$ over the column axis,
the $x\in I_c$ slice integrates $|\tau(\cdot,\hat x)|^\alpha$ over the row
axis). Hence $\sigma_\tau\in\mathcal F_\alpha$, and it is
$\mathrm{PC}$-equivalent iff $\tau$ is on both axes.

**Step 2 (invoke `weakening`).** Apply Theorem `weakening` to the
$(N+M)$-square symmetric matrix $\bar A_{N+M}=a_{N+M}^{-1}\bar Y^{\sigma_\tau}$.
It yields a symmetric limit $\mu^{\sigma_\tau}$ with
$G_{\sigma_\tau}(z)=\tfrac1z\int_0^1 h_\alpha(Y^{\sigma_\tau}_v(z))\,dv$ and,
for $|z|\ge R$,
$z^\alpha Y^{\sigma_\tau}_x(z)=C_\alpha\int_0^1|\sigma_\tau(x,v)|^\alpha
g_\alpha(Y^{\sigma_\tau}_v(z))\,dv$, with the convergence almost sure under the
$\mathrm{PC}$-equivalence of Step 1 and in expectation otherwise. Uniqueness
and analyticity are those of `weakening`.

**Step 3 (block split).** Since $\sigma_\tau$ vanishes on the diagonal blocks,
the integral for $x\in I_r$ sees only $v\in I_c$ and conversely. Set
$Y_r(\hat x,z)=Y^{\sigma_\tau}_x(z)$ for $x\in I_r$ and
$Y_c(\hat y,z)=Y^{\sigma_\tau}_y(z)$ for $y\in I_c$. The change of variables
$dv=\Delta_c\,d\hat v$ on $I_c$ and $dw=\Delta_r\,d\hat w$ on $I_r$ turns the
`weakening` field equation into the pair (ii), with the prefactors
$C_\alpha\Delta_c=\tfrac{\gamma}{1+\gamma}C_\alpha$ and
$C_\alpha\Delta_r=\tfrac1{1+\gamma}C_\alpha$.

**Step 4 (the sum-rule -- the load-bearing step).** Multiply the $Y_r$-equation
by $g_\alpha(Y_r(x,z))$ and integrate $\Delta_r\!\int_0^1 dx$; multiply the
$Y_c$-equation by $g_\alpha(Y_c(y,z))$ and integrate $\Delta_c\!\int_0^1 dy$.
The two right-hand sides are
$$
\Delta_r\tfrac{\gamma}{1+\gamma}C_\alpha
\!\iint|\tau(x,v)|^\alpha g_\alpha(Y_r(x))g_\alpha(Y_c(v))\,dv\,dx,
\qquad
\Delta_c\tfrac1{1+\gamma}C_\alpha
\!\iint|\tau(w,y)|^\alpha g_\alpha(Y_c(y))g_\alpha(Y_r(w))\,dw\,dy,
$$
which are **identical**: relabel $(w,y)\mapsto(x,v)$ in the second, and note
$\Delta_r\tfrac{\gamma}{1+\gamma}=\Delta_c\tfrac1{1+\gamma}
=\tfrac{\gamma}{(1+\gamma)^2}$. Equating the left-hand sides and cancelling
$z^\alpha$ gives the sum-rule
$$
\Delta_r\big\langle Y_r\,g_\alpha(Y_r)\big\rangle
=\Delta_c\big\langle Y_c\,g_\alpha(Y_c)\big\rangle .
$$
Using $h_\alpha=1-\tfrac\alpha2 y g_\alpha(y)$ termwise and
$\Delta_r+\Delta_c=1$,
$$
zG_{\sigma_\tau}(z)
=\Delta_r\langle h_\alpha(Y_r)\rangle+\Delta_c\langle h_\alpha(Y_c)\rangle
=\frac{2}{1+\gamma}\,\langle h_\alpha(Y_r)\rangle-\frac{1-\gamma}{1+\gamma}.
$$

**Step 5 (bipartite symmetrization).** The Hermitization
$\bar Y^{\sigma_\tau}$ has eigenvalues $\pm s_k$ ($s_k$ the singular values of
$a_{N+M}^{-1}X_{N,M}$) and $N-M$ zeros, so (as in `hermitisation.md` eq. (10)) the
squared-singular law $\nu$ satisfies
$$
G_\nu(\zeta)=\frac{1+\gamma}{2\sqrt\zeta}\,G_{\sigma_\tau}(\sqrt\zeta)
+\frac{1-\gamma}{2\zeta},
$$
with the atom of mass $1-\gamma$ at $0$. Substituting Step 4 (with
$z=\sqrt\zeta$) cancels the $\tfrac{1-\gamma}{2\zeta}$ terms and yields (iii);
Stieltjes inversion gives (iv) and the boundedness off zero is inherited from
`weakening`.

**Step 6 (tail).** By `weakening`,
$t^{\alpha+1}\rho^{\sigma_\tau}(t)\to\tfrac\alpha2\iint|\sigma_\tau|^\alpha$,
and $\iint|\sigma_\tau|^\alpha=2\Delta_r\Delta_c\iint|\tau|^\alpha
=\tfrac{2\gamma}{(1+\gamma)^2}\iint|\tau|^\alpha$. Pushing the symmetric tail
through $s\mapsto s^2$ and the bipartite mass split gives (v). Corollary 1 is
the case $\tau\equiv1$, where the $x$-independence collapses $\langle\cdot
\rangle$ and the system becomes the two scalars of `wishart-amir`. $\qquad\square$

---

## Theorem 2 (One-sided profile: scalar reduction and density)

*Adopt the setup of Theorem 1 together with the additional hypothesis that the
row field is constant in $x$, $Y_r(x,z)\equiv Y_r(z)$. A clean, robust,
checkable sufficient condition is that $\tau$ be **one-sided**:
$|\tau(x,y)|=c(y)$ depends on the column coordinate only. (Canonical case:
column-scaling; $c\equiv\mathrm{const}$ is plain Wishart; the row-only
profile $|\tau|=r(x)$ follows by the transpose
$X_{N,M}\leftrightarrow X_{N,M}^{\mathsf t}$, with $\gamma\mapsto1/\gamma$ and
the zero-atom rebooked.) The hypothesis is what makes the Theorem 1(iii)
collapse a **single-scalar** object; everything below flows from that.*

**(i) (Reduced field).** *$Y_c$ is explicit, slaved to the single scalar
$Y_r$,*
$$
Y_c(v,z)=\frac{C_\alpha\,c(v)^\alpha}{(1+\gamma)\,z^\alpha}\,
g_\alpha\big(Y_r(z)\big),
$$
*and $Y_r:\mathbb C^+\to\mathcal K_\alpha$ is the unique analytic solution,
$\to0$ at infinity, of the scalar (nested) closure*
$$
z^\alpha Y_r=\frac{\gamma}{1+\gamma}C_\alpha\int_0^1 c(v)^\alpha\,
g_\alpha\!\Big(\frac{C_\alpha c(v)^\alpha}{(1+\gamma)z^\alpha}\,
g_\alpha(Y_r)\Big)\,dv .
$$

**(ii) (Transform and density).** *The Theorem 1(iii) collapse is a
single-scalar transform,*
$$
G_\nu(\zeta)=\frac1\zeta\,h_\alpha\big(Y_r(\sqrt\zeta)\big),
$$
*so $\mu_\alpha^{\gamma,\tau}$ has the atom $1-\gamma$ at $0$ and continuous
density, bounded off zero, on $(0,\infty)$*
$$
\rho_\alpha^{\gamma,\tau}(t)=-\frac1{\pi t}\,\Im\,h_\alpha\big(Y_r(\sqrt t)\big),
$$
*which, inherited from `weakening`, is real-analytic on $(R,\infty)$ for some
finite $R=R_\alpha^{\gamma,\tau}$ (the same threshold beyond which $Y_r$
continues analytically). Non-vanishing of the density near $0$ is **not
asserted for a general one-sided profile** -- it holds only in the degenerate
corner (see Specializations and remarks).*

For a general two-sided $\tau$ this Theorem 2 does not apply: $Y_r$ stays
functional and only Theorem 1's field-level forms hold.

### Proof of Theorem 2

**(i).** One-sidedness gives $|\tau(x,v)|^\alpha=c(v)^\alpha$, independent of
$x$, so the right-hand side of the Theorem 1(ii) row equation is $x$-free for
every $z$ and every solution $Y_c$; hence $Y_r(x,z)$ is constant in $x$ -- a
scalar -- unconditionally (no fragile cancellation, no dependence on $\gamma$ or
on the solution). Substituting the scalar $Y_r$ into the Theorem 1(ii) column
equation, $z^\alpha Y_c(y)=\tfrac{C_\alpha}{1+\gamma}g_\alpha(Y_r)
\int_0^1|\tau(w,y)|^\alpha dw=\tfrac{C_\alpha}{1+\gamma}g_\alpha(Y_r)\,c(y)^\alpha$
(using $\int_0^1 c(y)^\alpha dw=c(y)^\alpha$), the displayed slaved $Y_c$.
Re-inserting it into the row equation gives the nested scalar closure.

**(ii).** With $Y_r$ scalar, $\langle h_\alpha(Y_r)\rangle=h_\alpha(Y_r)$, so
Theorem 1(iii) reads $G_\nu(\zeta)=\tfrac1\zeta h_\alpha(Y_r(\sqrt\zeta))$;
Stieltjes inversion gives the density, and the atom $1-\gamma$, boundedness off
zero, and real-analyticity on $(R,\infty)$ for the finite threshold
$R=R_\alpha^{\gamma,\tau}$ (beyond which $z\mapsto Y_r$ continues analytically)
are inherited from `weakening` / Theorem 1. $\qquad\square$

---

## Specializations and remarks

- **One-sided is the Theorem 2 hypothesis, not a specialization.**
  $|\tau(x,y)|=c(y)$ (or, by the transpose, $r(x)$) is exactly the scalar-$Y_r$
  condition; the structured solver collapses to this scalar automatically
  (validated by `compare_one_sided_to_scalar_closure`). Plain Wishart
  `wishart-amir` ($c\equiv\mathrm{const}$, Corollary 1) and, at $\gamma=1$,
  the Wigner-Levy law are its degenerate corner (next bullet).

- **Degenerate corner ($c\equiv\mathrm{const}$).** One-sided with constant $c$
  is plain Wishart `wishart-amir`; at $\gamma=1$ it is the Wigner-Levy law
  $\mu_\alpha$ via $\rho^1_\alpha(t)=2^{1/\alpha}t^{-1/2}
  \rho_\alpha(2^{1/\alpha}\sqrt t)$ (Corollary 1). There the closure is the
  genuine single fixed point $z^\alpha Y=C_\alpha g_\alpha(Y)$ (`eq:BAG6`), so
  the density takes the algebraic form
  $\rho_\alpha(t)=\tfrac{\alpha}{2\pi}\Im[C_\alpha^{-1}|t|^{\alpha-1}Y(|t|)^2]$
  (from $g_\alpha(Y)=z^\alpha Y/C_\alpha$ in $h_\alpha=1-\tfrac\alpha2 Y
  g_\alpha$; both equalities of the paper's `rhoalpha`), is real-analytic on
  $(R,\infty)$, and **does not vanish in any neighbourhood of $0$** -- unlike
  the Marchenko-Pastur law $\mu_2^\gamma$, which vanishes throughout
  $[0,1-\gamma]$ (paper's Remark after `wishart-amir`). These qualitative
  features are **not** asserted for a general one-sided profile: `weakening`
  does not give the non-vanishing; only `gammaone` ($\sigma\equiv1$) does.

- **Two-sided $\tau$ is outside Theorem 2.** If $|\tau|$ depends on both
  coordinates, $Y_r$ stays functional; only Theorem 1 applies -- no scalar
  reduction.

- **Quadrature consistency (implementation).** The collapse in Step 4 and
  Theorem 2(ii) use the *analytic* identity $h_\alpha=1-\tfrac\alpha2 y
  g_\alpha$. Under finite Gauss-Laguerre quadrature this holds only if
  $h_\alpha$ is **defined** as $1-\tfrac\alpha2 y\,g_\alpha$ from the *same*
  rule; evaluating $h_\alpha$ and $g_\alpha$ as independent Laguerre sums
  breaks it (numerically $\sim5\times10^{-3}$ at order $64$-$128$, while the
  shared-rule sum-rule $\Delta_r\langle Yg_\alpha(Y)\rangle=\Delta_c\langle
  Y_cg_\alpha(Y_c)\rangle$ holds to $\sim10^{-14}$). `structured_wishart_levy.py`
  uses the shared-rule $h_\alpha$ throughout.

- **Deterministic $\tau$ only.** A random multiplicative profile (e.g. the DNN
  Jacobian column scaling $|\phi'(h_j)|$) is outside the scope of `weakening`;
  the correct reduction is the deterministic quantile embedding of the
  mean-field $|\phi'|$ law -- replace the random column scaling by the
  deterministic quantile function of its limiting distribution, then apply
  Theorem 2. The paper's
  Theorem `weakeningD` (an independent *additive* diagonal of finite second
  moment) is not the right machinery for a multiplicative $\tau$.

- **Convergence mode.** Almost-sure convergence requires $\tau$ to be
  piecewise-constant-equivalent on both axes (Step 1); otherwise Theorem 1(i)
  is the convergence of $\mathbb E[\hat\mu_{W_{N,M}}]$.

- **Normalization (do not skip).** Theorems 1-2 deliver the **squared**,
  atom-carrying law $\nu$ (density of (ii), mass $\gamma$ off the $1-\gamma$
  atom). To compare with singular values one must pass through the
  Preliminaries bridges: an empirical SVD histogram is the atomless $M$-value
  law $\mu_{\mathrm{SV}}$, whereas `structured_wishart_levy.py` theory curves
  are the Gram law $\mu^{(N)}_{\mathrm{SV}}$ (mass $\gamma$, atom $1-\gamma$);
  the two differ by a factor $\gamma$. Conflating them is the spurious
  "$\sim(1-\gamma)$ discrepancy".

## Summary

**Theorem 1** is the general structured law: it follows from the continuum
general profile theorem `weakening` by the same Hermitize-specialize-
pushforward recipe as `wishart-amir`, the only novel algebra being the sum-rule
(Step 4, by relabelling) that collapses the coupled $(Y_r,Y_c)$ system to
$G_\nu(\zeta)=\tfrac1\zeta\langle h_\alpha(Y_r(\cdot,\sqrt\zeta))\rangle$;
setting $\tau\equiv1$ returns Theorem `wishart-amir` verbatim, so it slots into
the paper as a strict generalization.

**Theorem 2** adds the one hypothesis that the row field is scalar -- cleanly,
robustly, and checkably guaranteed by a **one-sided profile** $|\tau|=c(y)$
(the column-scaling case; $c\equiv\mathrm{const}$ is plain Wishart). Then the
transform is the single-scalar $G_\nu=\tfrac1\zeta h_\alpha(Y_r)$ and the
singular-value density is $-\tfrac1{\pi t}\Im h_\alpha(Y_r(\sqrt t))$ plus the
$1-\gamma$ atom. The degenerate Wigner/plain-square corner additionally
inherits the algebraic density and the qualitative features of `wishart-amir`.
Two-sided profiles fall outside Theorem 2 and are governed by Theorem 1 alone
(whose tail law (v) already gives the one-sided $s^{-1-\alpha}$ asymptotic with
constant $\int_0^1 c^\alpha$).
