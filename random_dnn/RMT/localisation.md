# Localisation and the mobility edge for structured heavy-tailed matrices

A general RMT foundation: the Anderson localisation transition (mobility edge)
for structured heavy-tailed random matrices $A = (a_{ij} x_{ij})$, with
$x_{ij}$ i.i.d. Levy $\alpha$-stable and $a_{ij}$ an arbitrary deterministic
entry profile. The method is the Tarquini-Biroli-Tarzia (TBT) transfer-operator
analysis adapted to the Belinschi structured ensemble. A constant profile
recovers TBT verbatim. The heavy-tailed MLP Jacobian is one downstream
specialisation (Section 6, a Gate), not part of the core.

Companion files:
- `structured_wishart_levy.md` -- the real-part / spectral-density closure; the
  analytic input reused in Section 2.
- `hermitisation.md` -- bipartite Hermitisation of the singular-value problem.
- `levy_mobility_edge.md` -- the unstructured symmetric anchor (TBT closed
  mobility-edge equation), against which Section 4 reduces and validates.
- literature notes: `.agents/notes/tarquini-2015.md`, `belinschi-2008.md`,
  `bordenave-2012.md`, `cizeau-bouchaud-1994.md`.

---

## 0. Why the density is analytic and localisation is not

The two questions one asks of the resolvent $G_{ii}(z) = [(z-H)^{-1}]_{ii}$ at
$z = E + i\eta$ are computed from genuinely different functionals of its
limiting law.

- **Spectral density** is $\rho(E) = -\frac{1}{\pi}\langle \mathrm{Im}\,G\rangle$:
  a *linear* (first-moment) functional of the resolvent law. The
  Belinschi-Dembo-Guionnet (BDG) limit hands over the full limiting law of $G$
  as a complex $\alpha/2$-stable distribution, so $\langle\mathrm{Im}\,G\rangle$
  is a single integral. Closed form (`structured_wishart_levy.md`, Thm 1(iv)).

- **Localisation** is *not* a moment of that law at all. It is the **stability of
  the $\mathrm{Im}\,G = 0$ fixed point** under the cavity recursion. Linearising
  (Section 3) makes this explicit: the imaginary part obeys
  $\Delta_i = \sum_j \frac{a_{ij}^2 x_{ij}^2}{(E - S_j)^2}\,\Delta_j$, and the
  bond weight $\frac{1}{(E-S)^2} = |\mathrm{Re}\,G|^2$ is the squared modulus of
  the resolvent. The transition is governed by fractional moments of $|G|$, not
  of $\mathrm{Im}\,G$ -- the *width* of the law, not its mean. This has no
  single-integral closed form.

TBT's achievement is that one need not compute the non-analytic $|G|$-moments
to *locate the transition*: the transfer-operator stability closes via a
directed-polymer freezing analysis that collapses to a replica index $m = 1/2$.
That is the only tractable route to the $|G|$-localisation object, and it is the
spine of this note.

Asymptotic corollary (TBT): the transition is direct (delocalised/localised);
the multifractal exponents satisfy $D_q \in \{0, 1\}$ in the $N \to \infty$
limit, so a non-trivial $D_q(s)$ is a **finite-$N$ crossover** phenomenon
(Section 5), not an asymptotic multifractal phase. Locating the mobility edge is
therefore the complete asymptotic localisation answer.

---

## 1. Setting: structured ensemble and bipartite cavity

Let $A \in \mathbb{R}^{N \times M}$, $A_{ij} = a_{ij}\, x_{ij}$, with $x_{ij}$
i.i.d. symmetric Levy $\alpha$-stable (tail index $\alpha \in (0, 2)$) scaled so
the bulk singular values are $O(1)$, and $a_{ij} \ge 0$ a deterministic profile
with empirical two-dimensional law $\Pi$. The singular-value problem is the
eigenvalue problem of the bipartite Hermitisation
$H = \begin{pmatrix} 0 & A \\ A^\top & 0 \end{pmatrix}$ (`hermitisation.md`):
the eigenvalues are $\pm s_k$, and at $z = E + i\eta$ the bipartite coordinate
$E$ coincides with the singular value $s$ for $E > 0$. Left/right singular
vectors are the row-block / column-block restrictions of the eigenvectors, so
**row-side localisation = left-SV localisation, column-side = right-SV**.

On the locally tree-like bipartite graph the diagonal cavity resolvents split
into a row side $G^R_i$ and a column side $G^C_j$:
$$
G^R_i = \Big(z - \sum_{j} a_{ij}^2\, x_{ij}^2\, G^C_j\Big)^{-1},
\qquad
G^C_j = \Big(z - \sum_{i} a_{ij}^2\, x_{ij}^2\, G^R_i\Big)^{-1}. \tag{1}
$$
The profile $a_{ij}$ sits on both sides but is summed over different indices,
which is what breaks the row/column (left/right) symmetry.

---

## 2. Real-part closure: the spectral density (analytic, reused)

Write the self-energy $\Sigma^R_i = z - (G^R_i)^{-1} = S^R_i + i\Delta^R_i$.
As $\eta \to 0^+$ and to linear order in $\Delta$, the real part decouples:
$$
S^R_i = \sum_j a_{ij}^2\, x_{ij}^2\, \mathrm{Re}\,G^C_j,
\qquad \mathrm{Re}\,G^C_j = \frac{1}{E - S^C_j}. \tag{2}
$$
Since $x^2$ is heavy-tailed with index $\alpha/2$, the generalized CLT makes
$S^R_i$ a Levy-stable variable of index $\alpha/2$, whose scale and skewness
depend on the row profile $\{a_{ij}\}_j$:
$$
C^R_i = \Gamma\!\big(1-\tfrac\alpha2\big)\cos\tfrac{\pi\alpha}{4}\cdot
\frac1N\sum_j a_{ij}^{\alpha}\,\mathbb{E}\big|\mathrm{Re}\,G^C_j\big|^{\alpha/2},
\qquad
\beta^R_i = \frac{\sum_j a_{ij}^{\alpha}\,
\mathbb{E}\big[\mathrm{sgn}(\mathrm{Re}\,G^C_j)|\mathrm{Re}\,G^C_j|^{\alpha/2}\big]}
{\sum_j a_{ij}^{\alpha}\,\mathbb{E}|\mathrm{Re}\,G^C_j|^{\alpha/2}}, \tag{3}
$$
(using $(a_{ij}^2)^{\alpha/2} = a_{ij}^{\alpha}$), with the mirror relations on
the column side. This coupled closure over the profile law $\Pi$ **is** the
structured-Wishart-Levy real-part fixed point of `structured_wishart_levy.md`
(Theorem 1); from it the density follows by the imaginary part as in Section 0.
It is the analytic input to the localisation analysis -- not re-derived here.

For a one-sided row profile $a_{ij} = a_i$ the factor $a_i^{\alpha}$ pulls out of
(3) and the row-side closure factorises through the single average
$\langle a^{\alpha}\rangle$; for a constant profile it reduces to the scalar TBT
$(C, \beta)$ of `levy_mobility_edge.md`.

---

## 3. Imaginary-part recursion: TBT's derivation, carrying the profile

This section follows the TBT EPAPS ("Computation of the mobility edge") step by
step, inserting the entry $A_{ij} = a_{ij} x_{ij}$ and tracking where the profile
appears. The entry $x_{ij}$ has tail $P(x) \simeq (\alpha/2N)|x|^{-1-\alpha}$;
$a_{ij} \ge 0$ is deterministic. TBT's parameter is the entry index, so the
substitution is $\mu \to \alpha$ together with $h_{ij} \to a_{ij} x_{ij}$.

### 3.1 Linearised recursion (TBT eq:S, eq:Delta)

Linearising (1) in the imaginary part about the real-part fixed point (2), with
$\mathrm{Im}\,G^C_j = \Delta^C_j/(E - S^C_j)^2$,
$$
\Delta^R_i = \sum_j \frac{a_{ij}^2\, x_{ij}^2}{(E - S^C_j)^2}\,\Delta^C_j,
\qquad
\Delta^C_j = \sum_i \frac{a_{ij}^2\, x_{ij}^2}{(E - S^R_i)^2}\,\Delta^R_i. \tag{4}
$$
The bond weight $w_{ij} = a_{ij}^2 x_{ij}^2 |\mathrm{Re}\,G|^2$ is the $|G|^2$
object of Section 0.

### 3.2 Directed polymer and the $m = 1/2$ freezing (TBT eq:DPRM, eq:phi)

Iterating (4), $\Delta_{i_1} = \sum_{\mathcal P}\prod_{\text{edges}} w \cdot
\Delta_{i_R}$ -- a **bipartite directed polymer** with quenched bond disorder
$w$. Localised = frozen (one-step RSB) phase ($\Delta$ decays under iteration);
delocalised = ergodic phase ($\Delta$ grows); the mobility edge $E^\star$ is the
freezing transition, $\phi(m^\star,E^\star)=0,\ \partial_m\phi=0$. As in TBT this
is the top eigenvalue $\lambda(m,E)$ of a transfer integral operator,
$\lambda(m^\star,E^\star)=1,\ \partial_m\lambda=0$.

### 3.3 The disorder integral: where the profile enters (TBT eq:924)

The transfer operator's kernel is built from the single-edge disorder integral
(TBT EPAPS eq. above their eq:final). With $h = a x$, $x$ of index $\alpha$:
$$
\int_0^\infty\! dx\,P(x)\,|a x|^{2m}\,e^{-ik a^2 x^2/(E-X')}
= a^{2m}\!\int_0^\infty\! dx\,P(x)\,|x|^{2m}\,e^{-i(k a^2) x^2/(E-X')}.
$$
Applying TBT's closed form (their eq:924, with $k \to k a^2$, $\mu \to \alpha$)
and $|k a^2/(E-X')|^{\alpha/2-m} = a^{\alpha-2m}|k/(E-X')|^{\alpha/2-m}$,
$\mathrm{sign}(k a^2(\cdot)) = \mathrm{sign}(k(\cdot))$, the two profile powers
combine:
$$
\boxed{\;a^{2m}\cdot a^{\alpha-2m} = a^{\alpha}\;}
\qquad\Longrightarrow\qquad
\int_0^\infty\! dx\,P(x)\,|a x|^{2m}\,e^{-ik a^2 x^2/(E-X')}
= a^{\alpha}\cdot \big[\text{TBT eq:924}\big]_{\mu\to\alpha}. \tag{5}
$$
**The profile enters every disorder integral as a factor $a^{\alpha}$ that is
independent of the replica index $m$.** This is the load-bearing structured fact:
because $a^\alpha$ does not depend on $m$, it does not break the reflection
symmetry of the determinant about $m = 1/2$ -- so the $m = 1/2$ collapse survives
the profile, and no separate verification of the stationary point is needed.
(The earlier guess that the profile enters as $\langle a\rangle$ at $m=1/2$ was
wrong; the correct, $m$-independent factor is $a^{\alpha}$.)

### 3.4 Structured mobility-edge determinant (TBT eq:final -> eq:mobility)

Carrying $a^\alpha$ through TBT's eq:final, the inverse-Fourier step, and the
$2\times2$ reduction $(I_+, I_-)$ reproduces TBT's algebra with two substitutions:
(i) the path-edge factor $a^{\alpha}$ rides on each transfer-operator application,
i.e. on each power of $K_\alpha = \tfrac{\alpha}{2}\Gamma(\tfrac12-\tfrac\alpha2)^2$;
(ii) the self-energy characteristic function $\hat L_{\alpha/2}^{C,\beta}$ carries
the **profile-weighted** $(C, \beta)$ of (3). The determinant at $m = 1/2$ is
TBT's with $K_\alpha \to K_\alpha\,\langle a^{\alpha}\rangle$:
$$
\boxed{\;
\big(K_\alpha \langle a^{\alpha}\rangle\big)^2\big(s_\alpha^2 - 1\big)|\ell|^2
- 2\,s_\alpha\,K_\alpha \langle a^{\alpha}\rangle\,\mathrm{Re}\,\ell + 1 = 0,
\;}
\tag{6}
$$
$s_\alpha = \sin(\pi\alpha/2)$, and $\ell(E) = \frac1\pi\int_0^\infty
k^{\alpha-1}\hat L_{\alpha/2}^{C(E),\beta(E)}(k)\,e^{ikE}\,dk$ with the
profile-weighted $(C, \beta)$. At $\Pi = \delta_{a=1}$ ($\langle a^\alpha\rangle =
1$, $(C,\beta)$ scalar) this is *exactly* the TBT equation of
`levy_mobility_edge.md` at $\mu = \alpha$ -- consistent with Section 4.

### 3.5 What (6) still elides (the genuine remaining work)

Equation (6) is the **mean-field / row-symmetric** form. Two structural details
are folded into the single average $\langle a^\alpha\rangle$ and must be unfolded
for the general profile:

1. **Profile resolution.** $\langle a^\alpha\rangle$ is an average over the
   path-edge profile, but the path-edge profile is correlated with the node
   self-energy $S$ (both depend on the source node's row of $\{a_{ij}\}$).
   For general $a_{ij}$ the transfer operator acts on functions of $(X, \text{
   source-profile})$ and (6) becomes a small integral equation over the profile
   coordinate, not a scalar. It collapses to (6) for a one-sided row profile
   $a_{ij}=a_i$ (then $\langle a^\alpha\rangle = \int a^\alpha\,d\Pi$ and $(C,\beta)$
   factor through $\langle a^\alpha\rangle$), and to TBT for $a \equiv 1$.
2. **Row/column (left/right) asymmetry.** The bipartite two-leg composition uses
   the row-side and column-side profile averages separately; an asymmetric
   profile gives two edges $E^\star_{\mathrm{left}} \ne E^\star_{\mathrm{right}}$.
   Equation (6) is the symmetric reduction; the asymmetric kernel is the
   outstanding derivation.

Both reduce, at $a\equiv 1$, to the single validated unstructured edge of
Section 4. Item 1 is carried out for the one-sided row profile in 3.6.

### 3.6 One-sided row profile: explicit reduction

Take $a_{ij} = a_i = c(v_i)$, $v_i \sim U[0,1]$ (row nodes carry the profile,
column nodes do not). Then $a_i^2$ pulls out of the row self-energy, so
$S^R_i \mid a_i$ is $\alpha/2$-stable of scale $a_i^{\alpha} C^R$ and skew
$\beta^R$, while the column self-energy is profile-*averaged*, scale $C^C$, skew
$\beta^C$, with (from eq. 3)
$$
C^R = \Gamma(1-\tfrac\alpha2)\cos\tfrac{\pi\alpha}{4}\,
      \mathbb{E}|\mathrm{Re}\,G^C|^{\alpha/2},
\qquad
C^C = \Gamma(1-\tfrac\alpha2)\cos\tfrac{\pi\alpha}{4}\!
      \int_0^1\! dv\, c(v)^{\alpha}\,\mathbb{E}|\mathrm{Re}\,G^R_{(c(v))}|^{\alpha/2},
$$
a coupled scalar closure ($\mathrm{Re}\,G^C = 1/(E-S^C)$,
$\mathrm{Re}\,G^R_{(a)} = 1/(E - S^R_{(a)})$); $\beta^{R,C}$ analogously. This is
the one-sided real-part fixed point of `one_sided_wishart_levy.md`.

**Coupled transfer operators.** With $m = 1/2$ throughout (3.3), the two legs are
$$
Z^R_{(a)}(X) = N\!\int\! dx\,P(x)\,dS\,L^R_{(a)}(S)\,dX'\,Z^C(X')\,
  \delta\!\big(X - S - \tfrac{a^2 x^2}{E-X'}\big)\big|\tfrac{a x}{E-X'}\big|,
$$
$$
Z^C(X) = N\!\int_0^1\!\! dv\!\int\! dx\,P(x)\,dS\,L^C(S)\,dX'\,Z^R_{(c(v))}(X')\,
  \delta\!\big(X - S - \tfrac{c(v)^2 x^2}{E-X'}\big)\big|\tfrac{c(v) x}{E-X'}\big|,
$$
the $\int_0^1 dv$ averaging over the row profile of the incoming edge.

**Fourier reduction.** Each disorder integral gives the $m$-independent profile
factor of 3.3 (eq. 5): $a^{\alpha}$ on the row leg, $c(v)^{\alpha}$ on the column
leg. Following TBT's eq:final + inverse-Fourier + $2\times2$ steps verbatim, with
$I^{R,C}_{\pm} = \int_{\gtrless 0}\tfrac{dk}{\pi}\,e^{ikE}|k|^{(\alpha-1)/2}
\hat Z^{R,C}(k)$, the legs become
$$
I^R_{\pm,(a)} = a^{\alpha} K_\alpha\,\ell^R_{\pm,(a)}\,(s_\alpha I^C_\pm + I^C_\mp),
\qquad
I^C_\pm = K_\alpha\,\ell^C_\pm\!\int_0^1\! dv\, c(v)^{\alpha}
  (s_\alpha I^R_{\pm,(c(v))} + I^R_{\mp,(c(v))}),
$$
$K_\alpha = \tfrac\alpha2\Gamma(\tfrac{1-\alpha}{2})^2$, $s_\alpha = \sin\tfrac{\pi\alpha}{2}$,
and the profile-resolved $\ell$:
$$
\ell^R_{+,(a)}(E) = \tfrac1\pi\!\int_0^\infty\! k^{\alpha-1}
  e^{-a^{\alpha} C^R k^{\alpha/2}(1 - i\beta^R t\,)}\,e^{ikE}\,dk,
\quad
\ell^C_{+}(E) = \tfrac1\pi\!\int_0^\infty\! k^{\alpha-1}
  e^{- C^C k^{\alpha/2}(1 - i\beta^C t)}\,e^{ikE}\,dk,
$$
$t = \tan\tfrac{\pi\alpha}{4}$, $\ell_- = \ell_+^*$. Substituting the row leg into
the column leg, the **two edges of each interior row node** combine its path-edge
$a^{\alpha}$ with the same node's self-energy, giving the weight $c(v)^{2\alpha}$.
Define
$$
L^R_\pm(E) := \int_0^1 dv\, c(v)^{2\alpha}\,\ell^R_{\pm,(c(v))}(E).
$$
The result is a $2\times2$ system in $(I^C_+, I^C_-)$ whose solvability is the

**one-sided structured mobility edge:**
$$
\boxed{\;
K_\alpha^4\,|\ell^C|^2|L^R|^2\,(s_\alpha^2-1)^2
- 2K_\alpha^2\Big[s_\alpha^2\,\mathrm{Re}\big(\ell^C L^R\big)
  + \mathrm{Re}\big(\ell^C\,\overline{L^R}\big)\Big] + 1 = 0.
\;}
\tag{7}
$$

**Constant-profile check.** At $c \equiv 1$: $C^R = C^C$, $\beta^R=\beta^C$, so
$\ell^C = \ell^R_{(1)} =: \ell$ and $L^R = \ell$; (7) factorises *exactly* as
$$
\big[K_\alpha^2(s_\alpha^2-1)|\ell|^2 - 2 s_\alpha K_\alpha\,\mathrm{Re}\,\ell + 1\big]
\cdot
\big[K_\alpha^2(s_\alpha^2-1)|\ell|^2 + 2 s_\alpha K_\alpha\,\mathrm{Re}\,\ell + 1\big] = 0,
$$
the first bracket being the TBT equation of `levy_mobility_edge.md` (the
$\lambda = +1$ freezing) and the second the spurious $\lambda = -1$ branch of the
two-leg map. **The physical edge is the TBT-connected root** (continued from
$c\equiv1$); for a non-constant profile (7) does not factor, and one tracks that
branch.

**Numerical recipe (solvable, like the unstructured kernel).** (i) Solve the
coupled $(C^R,\beta^R,C^C,\beta^C)$ closure by the deterministic $k$-space
fixed point of `levy_mobility_edge.py` generalised to the two-sided form;
(ii) build $\ell^C$, and $L^R$ by Gauss-Legendre over $v$ with the profile
$c(v)$ (one $\ell^R_{(c(v))}$ evaluation per node, reusing the QAWF integrator);
(iii) root-find (7) in $E$, tracking the physical (largest-magnitude / Perron)
branch. Reduction gate: at $c\equiv\text{const}$ and $\alpha < 1$ it must return
the `levy_mobility_edge.py` value -- **verified**
(`localisation.py` (Part 1): at $\alpha = 0.5$, $c\equiv1$, the
Perron edge is $3.29$, matching the unstructured solver; the two-sided closure
reproduces the symmetric $(C,\beta)$ to four digits).

**Domain of validity ($\mu < 1$ only).** Equation (7) -- like TBT's -- is built
from Fourier integrals $\Gamma(m-\mu/2)$, $\Gamma(1-m-\mu/2)$ whose convergence
needs $\mu < 1$; for $\mu \ge 1$ they are analytic continuations and TBT state
the equation "has a solution for $\mu \in (0,1)$ only". Numerically, the
$c\equiv1$ baselines at $\alpha = 1.5, 1.8$ return spurious sub-1 Perron
eigenvalues (false "edges" at $E \sim 1.2$) where Bordenave-Guionnet guarantee
delocalisation -- continuation artifacts, worsening as $\alpha$ moves above 1.
**So (7) cannot predict the operational-range ($\alpha \in (1,2)$) Jacobian
localisation -- but that is because (7) is the wrong mechanism, not because
there is no localisation.** Two distinct mechanisms must be separated:

1. *Heavy-tail-index localisation* (TBT/BG, eq. (7)): needs $\mu = \alpha < 1$.
   A bounded profile rescales $K_\alpha \to K_\alpha\langle c^\alpha\rangle$ but
   cannot change $\mu = \alpha$, so it is **absent for $\alpha \in (1,2)$** --
   correctly, per BG delocalisation.
2. *Profile-sparsification localisation* (structural, outside (7)): at the
   heavy-tailed fixed point the pre-activations $h$ are themselves $\alpha$-stable,
   so $\phi'(h) = \mathrm{sech}^2 h$ is near-zero on a finite fraction of
   saturated units -- effective row deletion / connectivity dilution, which
   localises by an Anderson/sparse mechanism, *not* via the heavy-tail index.

**The empirical Jacobian localised tail is mechanism 2** and is real, not a
crossover artifact. Direct surrogate test
(`.agents/temp/sv_localization_profiled.py`, $A = \mathrm{diag}(|\phi'(h)|)W$,
left-SV IPR $N$-scaling, $\alpha = 1.5$): as the saturated fraction grows
($s_h = 0 \to 0.5 \to 1.0$, i.e. $0 \to 5\% \to 18\%$ of $|\phi'| < 0.05$), the
tail $N$-scaling slope departs from the delocalised $-1$ ($-0.80 \to -0.53 \to
-0.38$ at $s = 2.9$) and the onset moves into the bulk ($s \approx 2.7 \to 2.1
\to 1.9$); additionally the *smallest* singular values localise (left vectors on
the $\phi'\approx0$ rows). So:

> The heavy-tailed MLP Jacobian tail localises through **profile
> sparsification** (saturated units), monotonically in the saturated fraction --
> not through the heavy-tail-index mobility edge, which is absent for
> $\alpha \in (1,2)$.

This localisation is **still the Tarquini / directed-polymer stability
criterion** (Anderson localisation on a tree *is* a DPRM freezing problem); it is
not a separate "sparse-RMT" framework. What changes for $\alpha \in (1,2)$ is
only that the criterion no longer has a closed-form solution -- it becomes an
integral-operator eigenvalue problem (3.7), equivalently the $\eta \to 0$
stability of $\mathrm{Im}\,G$ under the cavity RDE.

**It is a true asymptotic edge, not a finite-$N$ crossover.** Evaluating the
criterion directly by complex cavity population dynamics
(`localisation.py` (Part 2), $N \to \infty$ cavity) at $\alpha = 1.5$:
the $\eta$-scaling exponent $p = d\log\mathrm{Im}\,G_{\rm typ}/d\log\eta$ is
$p \approx 0$ at all $E$ unstructured (delocalised -- matches BG, validates the
tool), but with a saturating $\tanh$ row profile ($\sigma_h = 1.5$, 33% of units
with $|\phi'| < 0.05$) it rises through $1/2$ at $E^\star \approx 4.0$ and reaches
$p \to 1$ ($\mathrm{Im}\,G_{\rm typ} \propto \eta \to 0$, full localisation) in
the tail. So the profile induces a genuine **asymptotic mobility edge**
$s^\star \approx 4.0$ (at 33% saturation); the earlier IPR-surrogate ambiguity
(slopes $\sim -0.4$ at finite $N$) was just finite-$N$ -- the cavity shows the
transition is real. ReLU, $\phi' \in \{0,1\}$, is the exact-deletion limit of the
same mechanism.

### 3.7 Why (7) closes only for $\mu < 1$: scale-invariance, and the Bessel-kernel equation for $\alpha > 1$

For $\alpha > 1$ the path-edge disorder integral (3.3) is **bulk-dominated**
($\mathbb E|x| < \infty$) and must be taken against the full $\alpha$-stable
weight density $P_\alpha$, not its tail:
$$
I_\alpha(\omega) = \int P_\alpha(x)\,|x|\,e^{i\omega x^2}\,dx,
\qquad \omega = \frac{a^2 k}{E - X'},
$$
finite for $\alpha > 1$. The self-energy part ($\alpha/2$-stable, the $\ell$
integral) is unchanged -- only this entry integral breaks. Carrying $I_\alpha$
through TBT's $dX'$ step: with $y = E - X'$ and $y = \sqrt{q/p}\,e^t$, the inner
integral
$$
\int_0^\infty e^{i(py + q/y)}\,dy = \sqrt{q/p}\;g\!\big(2\sqrt{pq}\big),
\qquad g(z) = \int_{-\infty}^\infty e^t e^{i z \cosh t}\,dt \ \text{(Hankel-type)},
$$
is a prefactor times a function of the **product** $pq$ alone. Hence the
eigenvalue kernel is
$$
\mathcal K(k, k') \;\propto\; e^{-ik'E}\,\sqrt{k/k'}\;G_\alpha\!\big(\sqrt{k k'}\big),
\qquad G_\alpha(\xi) = \int P_\alpha(x)\,x^2\,g(2|x|\xi)\,dx.
$$
This factorises into $f(k)\,h(k')$ -- i.e. collapses to TBT's $2\times2$
determinant -- **iff $G_\alpha$ (equivalently $I_\alpha$) is a pure power**,
since $\xi^c = (kk')^{c/2}$ separates but a Bessel function does not. And
$I_\alpha$ is a pure power iff the disorder integral is **scale-free**, i.e.
tail-dominated, i.e. $\mu < 1$. Verified numerically (`.agents/temp`, FFT stable
density): $|I_\alpha(\omega)|$ has constant log-log slope (power law) at
$\alpha = 0.6$, but a slope sweeping $-0.3 \to -1.0$ through a crossover at
$\omega \sim 1$ at $\alpha = 1.5$ (the bulk scale) -- not a power law.

**So for $\alpha \in (1,2)$ the reduction does not close.** The mobility edge is
$$
\boxed{\ \text{top (Perron) eigenvalue of the integral operator with kernel }
\mathcal K(k,k') \ = \ 1\ }
$$
-- a 1-D Fredholm problem (Bessel kernel weighted by $P_\alpha$), solved by
discretising $k$; no closed transcendental. The closed form (7) is special to
the scale-invariance of the $\mu < 1$ heavy tail; removing it (bulk, $\alpha > 1$)
costs exactly one rung of closedness, leaving an integral operator rather than a
determinant. This is the correct tool for the profile-sparsification edge above.

---

## 4. Unstructured reduction, the index map, and the delocalisation baseline

Set $a_{ij} \equiv 1$. For the square case the bipartite cavity RDE is, by
row/column symmetry, **identical** to the symmetric heavy-tailed RDE of
Bordenave-Guionnet 2012 (`.agents/notes/bordenave-2012.md`),
$G = -(z + \sum_k \xi_k G_k)^{-1}$ with $\{\xi_k\}$ a Poisson process of intensity
$\propto \xi^{-1-\alpha/2}\,d\xi$. The squared-entry index $\alpha/2$ is already
internal to this RDE for *any* heavy-tailed symmetric or bipartite matrix; there
is **no additional Hermitisation halving**. Hence the index identification is
$$
\text{bipartite SV problem at entry index } \alpha
\;\equiv\; \text{TBT/BG symmetric problem at } \mu = \alpha,
$$
*not* $\mu = \alpha/2$. The Section-3 determinant must reduce to the scalar TBT
equation of `levy_mobility_edge.md` at this identification.

**Consequence (proven, Bordenave-Guionnet 2012):**
- $1 < \alpha < 2$: **delocalisation**. No localised phase; eigenvectors satisfy
  $\|v\|_p \to 0$ for all $p > 2$ off a finite exceptional set. So the
  **unstructured SV problem has no asymptotic mobility edge in the operational
  range** $\alpha \in (1, 2)$, and `levy_mobility_edge.py` correctly returns no
  edge for $\mu = \alpha > 1$.
- $\alpha < 2/3$: localisation in the spectral tail (proven). The window
  $2/3 \le \alpha \le 1$ is open (the Bouchaud-Cizeau threshold).

The direct IPR $N$-scaling (`.agents/temp/sv_localization_unstructured.py`:
delocalised slope $-1$ out to $s \approx 2.5$, only a weak departure in the far
heavy tail) and the density-deviation diagnostic onset $s_c$ are therefore
**finite-$N$ crossover** signatures (BG's exceptional set / TBT's wide-crossover
result 3), not the asymptotic edge. The $\alpha$-dependence of the apparent
onset does **not** distinguish a true edge from crossover, because the
heavy-tail outlier scale also shifts with $\alpha$.

**Implication for the structured problem.** Since the unstructured baseline is
delocalised for $\alpha \in (1, 2)$, any genuine asymptotic localisation in the
operational range must be **profile-induced**. A bounded profile
($|\phi'| \le 1$) does not change the entry tail index $\alpha$, so it cannot by
itself move the BG regime; it can only localise through effective
*sparsification* (a finite fraction of near-zero $\phi'$ -- saturated units) or
strong scale disorder. Whether this suffices for a true edge, or whether all
operational-range localisation is finite-$N$ crossover, is the load-bearing
question the structured determinant (Section 3) must answer.

**Validation gates:**
- **Gate A (done, this section):** index map $\mu = \alpha$, anchored in BG;
  the unstructured operational range is delocalised; `levy_mobility_edge.py`
  consistent. No finite edge to scale-convert.
- **Gate B (one-sided / row profile $a_{ij} = a_i$):** does a bounded row
  profile induce a *true* edge or only crossover? Decisive test: tail-$s$ IPR
  $N$-scaling on a row-scaled Levy matrix -- slope $\to 0$ with $N$ is a true
  edge, slope $\to -1$ is crossover.

---

## 5. Numerical scheme and the finite-$N$ diagnostic

Solving $D(E; \Pi) = 0$: iterate the structured real-part closure (3) to obtain
the profile-resolved $(C, \beta)$ field [the `structured_wishart_levy.py` /
`one_sided_wishart_levy.py` solvers]; build the profile-averaged kernel of
$\mathcal{T}_{1/2,E}$ and its Fourier reduction (Section 3, open step 2);
root-find in $E$. The unstructured kernel is already implemented and validated
in `levy_mobility_edge.py`.

Independently, the **density-deviation diagnostic** is a finite-$N$ probe of the
*onset* of localisation: where the population-dynamics estimate of the SV density
departs from the deterministic (BDG) theory marks the SV beyond which the
deterministic limit ceases to describe the typical local resolvent. It returns a
practical onset $s_c$ and the correct $\alpha$-trend, but -- per Section 0 -- it
measures a finite-$N$ crossover, not the asymptotic edge, and does not deliver a
closed $D_q$. Use it as a cross-check on $E^\star$, not as a substitute.

---

## 6. Downstream gate: the heavy-tailed MLP Jacobian

The layerwise Jacobian $J^l = D^l W^l$, $D^l = \mathrm{diag}(\phi'(h^l))$, is the
row-profile specialisation $a_{ij} = |\phi'(h^l_i)|$ of the structured ensemble.
The profile law $\Pi$ is the (quantile-embedded) distribution of $|\phi'|$ at the
heavy-tailed length-map fixed point of `heavy_tailed_mlp.md`; for $\tanh$ it
ranges from $\approx 0$ (saturated) to $\approx 1$ (linear). The structured
mobility edge then predicts the profile-induced shift of $s^\star$ relative to
the unstructured baseline of Section 4 -- i.e. whether saturated-unit structure
pushes the localisation onset into the operational bulk. The full derivation,
solver wiring, and comparison to a direct MLP-Jacobian IPR sweep live in
`ht_mlp_jacobian.{md,py,ipynb}`. This file states only the general result.

---

## Appendix: retracted approaches (do not repeat)

- **Profile-aligned mean-LDoS index $\ell_q$.** A Jensen gap of the deterministic
  per-position LDoS across the profile axis. It is a *density-variation* measure
  (first-moment / $\mathrm{Im}\,G$), identically zero for unstructured and for
  the row side of one-sided profiles, and does not detect intrinsic
  localisation. Superseded by the transfer-operator object of Section 3.
- **Direct closed-form $M_q = \mathbb{E}(-\mathrm{Im}\,G)^q$ via a spectral
  measure $\Gamma$.** An attempt to compute the non-analytic IPR moment directly
  from a self-consistency on the 2-D stable spectral measure. Stalled: the
  direct $|G|$-moment route is the hard one that TBT's $m = 1/2$ collapse
  sidesteps.
- **2-D Fourier pushforward of Belinschi's single-slice CF**; **one-ray ansatz
  for the resolvent law.** Both failed self-consistency numerically and were
  retracted.
