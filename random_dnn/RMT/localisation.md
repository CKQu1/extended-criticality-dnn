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

**The conclusion (a true asymptotic edge) survives, but every locator used here
does not.** *A true profile-induced edge does exist -- confirmed by the $N$-ladder
two paragraphs down (the structured tail $D_2$ stops delocalising between
$s \approx 5$ and $6.5$). But the typical-$p$ value $s_c \approx 4.0$ read off
below is an $\eta$-/finite-size artifact and not the locator; see the 2026-06
control and ladder.* Evaluating the
criterion directly by complex cavity population dynamics
(`localisation.py` (Part 2), $N \to \infty$ cavity) at $\alpha = 1.5$:
the $\eta$-scaling exponent $p = d\log\mathrm{Im}\,G_{\rm typ}/d\log\eta$ is
$p \approx 0$ at all $E$ unstructured (delocalised -- matches BG, validates the
tool), but with a saturating $\tanh$ row profile ($\sigma_h = 1.5$, 33% of units
with $|\phi'| < 0.05$) it rises through $1/2$ at $s_c \approx 4.0$ and reaches
$p \to 1$ ($\mathrm{Im}\,G_{\rm typ} \propto \eta \to 0$, full localisation) in
the tail. So the profile induces a genuine **asymptotic mobility edge**
$s_c \approx 4.0$ (entry-scale-1 units, at 33% saturation); the earlier IPR-surrogate ambiguity
(slopes $\sim -0.4$ at finite $N$) was just finite-$N$ -- the cavity shows the
transition is real. ReLU, $\phi' \in \{0,1\}$, is the exact-deletion limit of the
same mechanism.

**Caveat (2026-06): the typical $p$ is an unreliable witness for this
profile-induced edge.** The localisation here is the DPRM *freezing* problem
(above), and the order parameter $p = d\log\mathrm{Im}\,G_{\rm typ}/d\log\eta$
is the *typical* ($q \to 0$) member of the moment family -- the member that
freezing decouples from the resonance-dominated ($q \ge 1/2$) sector where the
localisation actually lives (sec. 3.9). For the
actual Jacobian profile ($\sigma_w = 3$, $\alpha = 1.5$) a deep-$\eta$ ladder
(`.agents/temp/d1_eta_ladder.py`) shows the structured population $p$ does **not**
plateau -- it *drifts toward $0$* (delocalised) at $s = 6.5$, never reaching
$p \to 1$. At $s = 6.5$ exact diagonalisation of the Hermitisation at $N = 3000$
(`.agents/temp/exact_resolvent.py`) gives, on the *same* matrix, eigenvector
$D_1 = 0.18$ and an exact typical-site $\mathrm{Im}\,G$ scaling $p \approx 0.93$
(localised-looking), vs $D_1 = 0.82$, $p \approx 0.25$ in the bulk. *This was
earlier read as confirming the asymptotic edge "by exact diagonalisation". The
unstructured control below shows that reading is unsupported* -- a single-$N$
$D_q$ in the tail is not a localisation signature.

**Control (2026-06): no single-$N$ witness locates an asymptotic edge in the
tail, because the BG-delocalised unstructured tail mimics all of them.** Running
the frozen tangent $\varphi^\star(s) = \min_{m\le\alpha/2}\log\lambda(m)/m$
(`.agents/temp/struct_vs_unstruct_edge.py`) on the *unstructured* Levy bond
(profile off, $a_i=1$) at $\alpha=1.5$ reports $\varphi^\star < 0$ ("localised")
for $s \gtrsim 4.5$ -- but BG guarantee $\alpha=1.5$ is delocalised everywhere
outside a finite exceptional set (tail localisation needs $\alpha < 2/3$). So the
frozen tangent *fails its own control*: it manufactures an edge where none
exists, and the structured $\varphi^\star=0$ at $s\approx7.2$ is the same artifact
shifted out by the profile, not an asymptotic locator. The same verdict reached
via exact diagonalisation, method-independently
(`.agents/temp/unstruct_exact_diag.py`, $N=1500\to3000$, 4 matrices each):
the *unstructured* singular-vector $D_2$ is already low in the tail at finite $N$
($D_2 = 0.58$ at $s{=}5$, $0.37$ at $s{=}6.5$, $0.12$ at $s{=}10$) -- yet its
$N$-trend is **positive in every bin** ($\mathrm dD_2/\mathrm d\log N \approx
+0.034$ at $s{=}5$ on 562 modes, up to $+0.065$), i.e. those states are slowly
*delocalising* with $N$, as BG require. Hence "low $D_q$ at one $N$" -- the basis
of the $D_1=0.18$ confirmation above and in 3.8-3.9 -- never distinguished
localisation from a delocalising tail.

What *is* robust and method-independent is the **level** contrast: the profile
suppresses $D_2$ by a large factor at every fixed $s$ (structured vs unstructured:
$0.51$ vs $0.79$ at $s{=}3$; $0.19$ vs $0.59$ at $s{=}5$; $0.09$ vs $0.37$ at
$s{=}6.5$). The profile-sparsification mechanism (3.6, saturated rows) drives
states toward localisation monotonically -- that is not in doubt. What the
single-$N$ snapshots **cannot** settle is whether the structured tail localises
*asymptotically* ($D_2 \to 0$) or is itself a slow finite-$N$ crossover: the
$1500\to3000$ lever (log-ratio $0.69$, 12-53 deep-tail modes) cannot separate
slope $0$ from the $+0.04$ baseline. Only the $N$-trend, on a real ladder with
filled bins, is a valid asymptotic discriminator (`.agents/temp/n_ladder_d2.py`,
$N=1000\dots6000$). So the order parameter is neither the typical $p$ nor the
frozen $\varphi^\star$ nor a single-$N$ $D_q$ -- it is the *$N$-scaling* of $D_2$
relative to the unstructured BG baseline; the surrogate $s_c\approx4.0$ and the
tangent $s_c\approx7.2$ are both discarded as locators.

**Resolved (2026-06) by the $N$-ladder, the valid discriminator: the profile DOES
induce a true edge -- the structured tail fails to delocalise.** Running that
discriminator (`.agents/temp/n_ladder_d2.py`, $N=1000,1500,2000,3000,4500,6000$,
up to 16 matrices per $N$ to fill tail bins; $D_2$ = mean singular-vector
correlation dimension per $s$-bin) gives, on the **identical $N$-window** (raw
per-$N$ means, no slope/SE modelling needed for the contrast):
| $s$ | unstructured $D_2$: $N{=}1000\to6000$ | structured $D_2$: $N{=}1000\to6000$ |
|----|----|----|
| 3.0 | $0.78 \to 0.82$ (climbs) | $0.48 \to 0.56$ (climbs) |
| 4.0 | $0.68 \to 0.74$ (climbs) | $0.27 \to 0.37$ (climbs) |
| 5.0 | $0.54 \to 0.63$ (climbs) | $0.17 \to 0.19$ (slow) |
| 6.5 | $0.35 \to 0.41$ (climbs) | $0.09 \to 0.09$ (**flat**) |
| 8.0 | $0.21 \to 0.28$ (climbs) | $0.05 \to 0.05$ (**flat**) |

The unstructured baseline climbs toward $D_2 = 1$ at *every* $s$ (BG
delocalisation, slow). The structured trend **crosses from climbing (bulk,
$s \lesssim 4$: low $D_2$ level but still delocalising, a finite-$N$ multifractal)
to flat (tail, $s \gtrsim 6.5$: $D_2$ stops drifting up)** between $s \approx 5$
and $s \approx 6.5$. A flat $D_2$ in the same $N$-window where the baseline
visibly climbs is the Gate-B signature (sec. 4) of a **genuine profile-induced
asymptotic edge** -- the tail states *fail to delocalise*, a real departure from
the BG baseline. Two honest limits: (i) the frozen tail $D_2 \approx 0.05$-$0.10$
is *not* drifting to $0$, so the claim is "does not delocalise", not "fully
localised" -- whether the frozen value is $0$ (Anderson) or a nonzero
critical/multifractal value (consistent with the DPRM *freezing* picture, 3.6) is
unresolved; (ii) the cleanest control -- matched-$D_2$-*level* slopes -- is
unavailable because the unstructured tail never reaches $D_2 \approx 0.1$ in
range, so this is a matched-$s$ contrast. The floor worry is dispatched in our
favour: a *delocalising* state with room above climbs (unstructured $s{=}8$,
$D_2{=}0.21 \to 0.28$); flatness near the floor is the freezing signature, not an
artifact. So a true edge exists; the gold point $s = 6.5$ sits *at or just past*
it (its tail $D_2$ does not delocalise) -- which is why the typical $p$ ($\to 0$),
the tangent ($\varphi^\star > 0$), and the single-$N$ $D_q$ all misread it.

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

### 3.8 Explicit $m$-resolved kernel (TBT EPAPS) and the $D_q$ slices

Section 3.7 gives the $m = 1/2$ (edge) kernel; the TBT EPAPS transfer operator
makes it explicit and $m$-resolved. The cascade (3.1) is $\Delta_i = \sum_j
[h_{ij}^2/(E - S_j)^2]\,\Delta_j$ (squared entry $\times$ squared neighbour
resolvent -- the directed-polymer edge weight). Its $m$-th-moment transfer
operator, in the frequency $k$ conjugate to the self-energy, acts as
$$
\hat Z(k) = N\,\hat L_{\alpha/2}^{C,\beta}(k)\int dX'\;|E - X'|^{-2m}\,
   I_\alpha^{(m)}\!\Big(\tfrac{k}{E-X'}\Big)\,Z(X'),
\qquad
I_\alpha^{(m)}(\omega) = \int P_\alpha(h)\,|h|^{2m}e^{-i\omega h^2}\,dh,
$$
the bond integral $I_\alpha^{(m)}$ (3.3) being the per-edge factor and
$|E-X'|^{-2m}$ the resolvent-resonance site factor. Closing with $Z(X') =
(2\pi)^{-1}\int\hat Z(k')e^{ik'X'}dk'$ and $y = E - X'$ gives the explicit 1-D
Fredholm kernel
$$
\boxed{\;
\mathcal K_m^E(k,k') = \frac{N}{2\pi}\,\hat L_{\alpha/2}^{C,\beta}(k)\,e^{ik'E}
   \int_{-\infty}^{\infty}|y|^{-2m}\,I_\alpha^{(m)}\!\Big(\tfrac{k}{y}\Big)\,
   e^{-ik'y}\,dy\;}
$$
with $\lambda(m,E)\,\hat Z(k) = \int\mathcal K_m^E(k,k')\hat Z(k')\,dk'$; the
$k/y$ scaling against $e^{-ik'y}$ is the origin of the $\sqrt{kk'}$ Bessel
structure of 3.7. One object, two slices:

- **Mobility edge:** $\lambda(\tfrac12, E^\star) = 1$. For $\alpha < 1$ the
  scale-free power-law eigenfunction diagonalises $\mathcal K$ and collapses it to
  the scalar $\ell$ of `levy_mobility_edge.md`; that collapse is what fails the
  $\mu = 1$ resonance-integrability test (3.7).
- **Multifractal spectrum:** the $m = q$ slice is $D_q$ -- $\lambda(q,E)$ is the
  deterministic counterpart of the $\eta$-scaling of $\langle(\mathrm{Im}\,G)^q
  \rangle$, so $\zeta_q = -\partial_{\log\eta}\log M_q$ is read off $\lambda(q,E)$
  and $D_q = 1 - \zeta_q/(q-1)$. This reaches the resonance-dominated $q > 1/2$
  regime the stochastic moments (3.9) cannot, and is the **freezing-correct**
  route for $\alpha > 1$.

*Build status (2026-06).* $I_\alpha^{(m)}(\omega)$ is implemented from the entry
CF (clean for $\alpha > 1$, the Jacobian regime; the $\alpha < 1$ scale-free cusp
needs graded quadrature). Open: the eigenvalue assembly -- the singular
$y$-integral (principal value at $y = 0$), the $\hat Z \leftrightarrow Z$ Fourier
closure, the $N$-normalisation pinned by the $\alpha < 1$ reduction to scalar
$\ell$, and the non-Hermitian top-eigenvalue solve. Until built and validated
against $E^\star(0.5) = 3.29$ and the $p = 1/2$ edge, the population dynamics
(Section 5) is the working edge -- with the 3.9 caveat.

*Build plan (M0-M4) and the M0 finding.* The kernel above is a **1-component
complex Fredholm** problem, and the boxed $\mathcal K_m^E$ follows *algebraically*
from the transfer-operator equation (substitute $Z(X') = (2\pi)^{-1}\int\hat
Z(k')e^{ik'X'}dk'$, then $y = E - X'$) -- no gap there. The TBT $2\times2$
determinant is **not** a separate two-component system: it rearranges to
$K_\mu^2(s^2-1)|\ell|^2 - 2sK_\mu\,\mathrm{Re}\,\ell + 1 = |sK_\mu\ell - 1|^2 -
|K_\mu\ell|^2$, so the edge $=0$ is the modulus condition $|sK_\mu\ell - 1| =
|K_\mu\ell|$ on the *single complex scalar* $\ell$ ($s = \sin(\pi\mu/2)$) -- and
$\ell$ is exactly the scale-free eigen-amplitude (power-law eigenfunction) of the
1-component kernel. So the $\alpha>1$ generalisation is the *same* complex
operator, the "$2\times2$" being only the real/imaginary structure of that
condition. The one unverified input is therefore the transfer-operator equation's
factors ($|E-X'|^{-2m}$, $I_\alpha^{(m)}$, $\hat L$), the normalisation $N$, and
the edge-coefficient convention ($s, K_\mu$; the box's "$\lambda=1$" is schematic,
the true condition carries the stable coefficients) -- **all pinned numerically by
M1**, so the reduction is a check, not a derivation blocker. Milestones, each
gating the next: **M0** (done) the structure above; **M1** the reduction gate --
assemble at $\alpha=0.5$, reproduce $E^\star(0.5)=3.29$ (validates the $y$-integral,
assembly, $N$, and coefficients at once); **M2** unstructured $\alpha=1.5 \to$ no
edge ($\min_m\lambda<1$, BG); **M3** profile $\to$ profile-induced edge, validated
against the exact-diag $N$-ladder (3.6: structured tail $D_2$ goes flat vs the
climbing BG baseline -- not the cavity $p$, the tangent $\varphi^\star$, or any
single-$N$ $D_q$, all of which misplace it); **M4** $D_q$ slices $+$ the
analytical SV density $\to$ a finite-size-free $c^\star$. Remaining risk now sits
in the **singular $y$-integral numerics** ($y\to0$ regularised by $I(k/y)\to0$,
plus the conditionally-convergent oscillatory tail) and the edge-coefficient
convention -- both settled at M1.

*M1 result (2026-06: PASSED).* The full chain is implemented and validated against
Tarquini's edge. Building blocks (in `.agents/temp/`): the bond integral
$I_\alpha^{(m)}(\omega)$ via the $u=h^2$ transform with a multi-term stable tail
(`bond_integral.py`; gates: $I(0)=\mathbb{E}|X|^{2m}$ to $10^{-4}$, scale-free slope
$(\alpha-1)/2$); $J_m(k,k')$ as a real half-line FFT (`jm_fft.py`; converges in
$dy,Y$); the kernel assembled on $k,k'>0$ with $(C,\beta)$ and $\hat L$ from
`levy_mobility_edge.py` (`kernel_assemble.py`/`kernel_loop.py`). Validated against
the **transfer-Perron oracle** `localisation.structured_perron` (which crosses $1$
at $E^\star=3.29$): the kernel's normalised physical eigenvalue crosses $1$ at
$E^\star\approx3.30$ (**$<1\%$**), tracking Perron to $\sim2\%$ (refined grid) /
$\sim7\%$ (coarse), grid-limited.

Two corrections to the M0 expectation, both forced by the numerics (do not repeat
the naive versions):
1. **Eigenvalue selection.** The discretised kernel has a *spurious* magnitude-largest
   branch; the physical mode is the **rank-1 scale-free power-law eigenfunction**
   ($\hat Z(k)\sim \hat L(k)\,k^{(\alpha-1)/2}$, effective slope $\approx-0.7$ at
   $\alpha=0.5$). Any $\alpha>1$ run must select by eigenfunction character, not by
   $|\lambda|$.
2. **The eigenvalue tracks the *Perron*, not $\ell$.** The tempting analytic
   simplification $\lambda=\text{const}\cdot\ell$ is **refuted** ($\lambda/\ell$
   varies $\sim20\%$ across $E$); the operator eigenvalue is the full (nonlinear)
   transfer Perron -- the $2\times2$ combination *of* $\ell$ -- and the
   normalisation is empirical, not the closed form a scale-free contraction would
   suggest. The M0 phrase "$\ell$ is the scale-free eigen-amplitude" is structurally
   right (the eigen*function* is the power law; $\ell$ is the scalar Perron is built
   from) but the eigen*value* is Perron, not $\ell$ itself.

So the machinery reduces correctly to Tarquini in the scale-free limit; the
$\alpha>1$ programme (M2-M4) is unblocked, with no quantitative oracle there -- it
is corroborated instead by the rigorous phase limits (BG delocalisation; Gaussian;
flat-Levy edge) and the validated analytical density. (Earlier this list also
named single-$N$ exact diagonalisation at the gold point $s=6.5$ -- $D_1=0.18$,
typical $p\approx0.93$ -- as corroboration; the 3.6 control retracts that: the
unstructured BG-delocalised tail has equally low single-$N$ $D_q$, so a single-$N$
$D_q$ is not an asymptotic-localisation witness. Only the $D_2$ $N$-trend relative
to the unstructured baseline is, 3.6.)

*M3 gate (2026-06): the $m=q$ slice yields the asymptotic binary $D_q$, not the
finite-$N$ multifractal value -- the route does not reproduce the empirical-style
$c^\star$.* Probed without the full kernel, via the cavity tangent (which gives
$\log\lambda(m,E)$ cleanly for $m<\alpha/2=0.75$, unlike the resonance moment
$M_q$; `.agents/temp/kernel_gate_lambda.py`) plus the exact-diag target
(`.agents/temp/kernel_gate_target.py`). Findings at the gold cell $(1.5,3.0)$:
(i) exact-diag (single-$N$ generalised dimensions, recipe of `cstar_grid.py`)
gives $D_1(s{=}6.5)=0.184$ at $N=3000$ -- reproducing the doc gold standard 0.18
-- and the target $D_2(s{=}6.5)=0.105$; it *grows* with $N$ ($D_2=0.062\to0.105$
over $N=1500\to3000$). (ii) The cavity $\log\lambda(m,E)$ is **linear in $m$ with
no kink at the freezing replica $m^\star=\tfrac12$** at every $s$, and
$\varphi^\star=\min_m\log\lambda/m$ crosses zero at $s\approx7.2$ (for this
$\sigma_w=3$ cell); the gold point $s=6.5$ sits *below* it ($\varphi^\star=+0.78$).
*This crossing was read as a "sharp asymptotic edge"; the 3.6 control + ladder
retract it as a locator -- the same tangent on the unstructured bond crosses at
$s\approx4.5$, where BG guarantee delocalisation, so $\varphi^\star$ manufactures
an edge wherever it sits and its location is not trustworthy. Note the tangent
$\varphi^\star=+0.78$ here reads the gold point as delocalised, but the valid
$N$-ladder discriminator (3.6) shows its tail $D_2$ does **not** delocalise --
i.e. the gold point is at/just past the true profile-induced edge, and the tangent
mislocates in the opposite direction to the typical $p$. The asymptotic edge is
real; only the cheap witnesses fail to place it.* Linear $\log\lambda(m)$
gives trivial $D_q$ under any bridge ($M_m\sim\eta^m\lambda(m)^L$; the candidate
$D_m = 1 - [\log\lambda(m)/\varphi_{\rm typ}-m]/(m-1)$ returns $D_m\equiv1$ for
$\log\lambda=\varphi_{\rm typ}m$). The single-$N$ $D_2=0.105$ is a finite-$N$
value -- *but the $N$-trend does **not** point to $D_q\to1$ at the gold point.*
The 2-matrix "grows $0.062\to0.105$ over $N=1500\to3000$" reported here was the
same 2-point artifact the 3.6 control exposes: the full ladder ($N=1000\dots6000$,
3.6) shows structured $s{=}6.5$ $D_2$ **flat** at $\approx0.09$
($0.094,0.082,0.092,0.107,0.114,0.094$) while the unstructured baseline climbs --
i.e. the gold point sits at/just past the profile-induced edge and does **not**
delocalise. So Section 0's binary $D_q\in\{0,1\}$ is the asymptotic *idealisation*;
the actual frozen tail $D_2\approx0.05$-$0.10$ may be exactly $0$ (Anderson) or a
nonzero critical value (DPRM freezing), which the ladder cannot yet separate (3.6).
A remaining kernel question -- whether $\log\lambda(m)$ stays linear out to $m=2$
or develops freezing convexity above the $m<0.75$ window -- needs the structured
kernel to read $\log\lambda(2)$; the in-window linearity through $m^\star=\tfrac12$
makes strong convexity unlikely, but the ladder's non-trivial frozen $D_2$ now
leaves room for mild convexity (a nonzero critical $D_2$), so this is no longer
settled against. **Consequence:** the kernel
route furnishes the asymptotic localisation *edge* (a sharp/binary $c^\star$), not
the smooth finite-$N$ multifractal $c^\star$; the latter is intrinsically
finite-$N$ and obtained only by diagonalisation (the empirical $c^\star$, or the
analytic-density $\times$ empirical-$D_1$ route, mean $|{\rm diff}|=0.08$).

### 3.9 $D_q$ from the cavity moments (stochastic readout, $q \le 1$)

The population dynamics also gives $D_q$ from its own moments, no kernel. With the
spectral resonance $\mathrm{Im}\,G_{ii} = \sum_\beta |\psi_\beta(i)|^2\,
\eta/((E-E_\beta)^2+\eta^2)$, averaging the $q$-th moment gives the moment-IPR
relation
$$
\boxed{\;\langle I_q\rangle = \frac{\eta^{q-1}}{\rho\,b_q}\,M_q(\eta)\;},
\qquad M_q(\eta) := \langle(\mathrm{Im}\,G)^q\rangle,\;\;
b_q = \sqrt\pi\,\frac{\Gamma(q-\tfrac12)}{\Gamma(q)}\ (q>\tfrac12),
$$
and with the tree dictionary $\eta \sim 1/N$ ($P_q \sim L^{-D_q(q-1)}$),
$$
\boxed{\;D_q(E) = 1 + \frac{1}{q-1}\,\frac{d\log M_q(\eta)}{d\log\eta}\;}
$$
(deloc: $M_q \to$ const, $D_q = 1$; loc: $M_q \sim \eta^{1-q}$, $D_q = 0$; the
$q \to 1$ member is $D_1 = 1 - p$). Implementation: accumulate $M_q(\eta) =
\mathrm{mean}((\mathrm{Im}\,G_r)^q)$ over the pool at two-or-more $\eta$ (the same
loop that forms $p$) and difference in $\log\eta$.

**Caveat (load-bearing for $\alpha > 1$): this uses the typical / low-$q$ members
and is the wrong slice in the frozen phase.** At the actual Jacobian profile
($\alpha = 1.5, \sigma_w = 3$) the population $p$ ($= 1 - D_1$) drifts to $0$
(delocalised) at $s = 6.5$ and never plateaus, while exact diagonalisation of the
Hermitisation at $N = 3000$ gives, on the *same* matrix, eigenvector $D_1 = 0.18$
and exact typical-site $\mathrm{Im}\,G$ scaling $p \approx 0.93$ (localised) -- vs
$D_1 = 0.82$, $p \approx 0.25$ in the bulk (`.agents/temp/d1_eta_ladder.py`,
`.agents/temp/exact_resolvent.py`). The earlier reading was that the stochastic
$D_{q\le1}$ *underestimates* the localisation and the single-$N$ exact-diag
$p\approx0.93$ is the truth. *Both halves are now corrected by the 3.6 control +
ladder, in different directions.* The single-$N$ $p\approx0.93$ / $D_1=0.18$ at
$s{=}6.5$ is **not** by itself an asymptotic localisation signature -- the
unstructured BG-delocalised tail carries equally low single-$N$ $D_q$. *However*
the $N$-ladder (3.6) shows the structured tail $D_2$ **does not delocalise**
(flat $\approx0.09$ at $s{=}6.5$ while the baseline climbs), so the gold point
*is* asymptotically non-delocalised, and the population $p\to0$ there **does**
underestimate -- it is the typical / low-$q$ witness reading the frozen tail as
delocalised, exactly the wrong slice. So the stochastic $p$ underestimates and the
single-$N$ exact-diag overstates *confidence* (its value is right but it cannot,
alone, distinguish a frozen tail from a delocalising one). What is genuinely
unreliable for $\alpha>1$ is *every single-$N$ / single-$\eta$ witness in the
tail* (typical $p$, frozen $\varphi^\star$, single-$N$ $D_q$); the only valid
asymptotic discriminator is the $D_2$ $N$-scaling against the unstructured
baseline (3.6). The typical-$p$/$D_{q\le1}$ readout remains correct for
$\alpha < 1$ (annealed = quenched) and as the delocalisation baseline
($p \approx 0$).

### 3.10 Finite-$N$ crossover line for $\alpha \in (1,2)$: the Thouless matching

For $\alpha \in (1, 2)$ Sections 3.7/4 leave an apparent paradox: the wide-$N$
spectrum is fully delocalised (no heavy-tail-index edge), yet finite-$N$
spectra -- exact diagonalisation, IPR $N$-scaling, and the cavity-vs-empirical
density deviation -- all show a localised-looking tail with an onset $s_c$
that *drifts* with the resolution ($N$, or the population size of the cavity).
TBT's result 3 names this regime (a wide finite-$N$ crossover) but derives no
crossover line for $\mu > 1$ (their EPAPS treats $\mu \in (1,2)$ only by Dyson
Brownian motion + numerics). This section derives the line $s_c(N)$ from the
delocalised fixed point itself. It is the quantitative content of the
"protocol-defined onset" of Section 5.

**Primary objects.** Both live on the delocalised side and are computable from
the cavity (population dynamics, or the 3.8 kernel):

1. $I_{\rm typ}(E) = \lim_{\eta\to0^+}\exp\langle\log \mathrm{Im}\,G\rangle$ --
   the stationary typical hybridisation width of the $N \to \infty$
   delocalised solution. Bordenave-Guionnet guarantee $I_{\rm typ} > 0$ for
   all $E$ at $\alpha \in (1,2)$ (`bordenave-2012.md`); in the spectral tail
   it is positive but decays steeply in $E$.
2. $\varphi^\star(E) = \min_m \tfrac1m\log\lambda(m, E)$ -- the frozen
   (1RSB-selected) Lyapunov exponent of the linearised $\mathrm{Im}$ recursion
   (4) about the real-part fixed point: the per-generation growth rate of an
   infinitesimal $\mathrm{Im}$ seed. $\lambda(m,E)$ is the 3.8 kernel's
   eigenvalue; $\varphi^\star$ is also directly measurable in population
   dynamics by seeding $\mathrm{Im}\,g = \epsilon$ and tracking
   $\langle\log\mathrm{Im}\rangle$ per sweep before saturation. For
   $\alpha > 1$: $\varphi^\star(E) > 0$ at every $E$ (that *is* the
   no-asymptotic-edge statement), with $\varphi^\star \to 0^+$ as
   $E \to \infty$.

**Criterion (Thouless matching).** At size $N$ the spectrum near $E$ is
discrete with mean level spacing $\Delta_N(E) = 1/(N\rho(E))$. The
infinite-size delocalised solution assigns states at $E$ a hybridisation
width $\Gamma(E) \sim I_{\rm typ}(E)$. Level mixing -- GOE statistics,
$\mathrm{IPR} \sim 1/N$, cavity density $=$ empirical density -- requires the
dimensionless Thouless ratio $g = \Gamma/\Delta_N = N\rho(E)\,I_{\rm typ}(E)
\gg 1$; for $g \ll 1$ each eigenstate at $E$ is an isolated resonance and the
sample is operationally localised there, even though $N \to \infty$
delocalises it. The crossover line is therefore
$$
\boxed{\;N\,\rho(s_c)\,I_{\rm typ}(s_c) = O(1).\;}
\tag{8}
$$
Every factor is known: $\rho$ analytically (`structured_wishart_levy.md`
Thm 1(iv), tail $f_{\rm SV}(s) \simeq B\,\langle\!\langle a^\alpha
\rangle\!\rangle\, s^{-1-\alpha}$ by Thm 1(v)); $I_{\rm typ}$ from
`localisation.py:cavity_typ_imG`. Since $\rho$ enters only through
$\log\rho$, the line is governed by the steep factor $I_{\rm typ}$.

**Near-marginal asymptotics of $I_{\rm typ}$ (hypothesis H1).** The
stationary $\mathrm{Im}$ law in the weakly delocalised regime is the
travelling-wave / front-selection fixed point of the linearised recursion
(the same DPRM structure as 3.2, now on the moving side of freezing): linear
growth at rate $\varphi^\star$ balanced by the saturation nonlinearity, with
the $m^\star = \tfrac12$ tail marginal. On the Bethe lattice this selection
gives (Mirlin-Fyodorov-type argument; transplanted here, not proven for the
fully-connected Levy case)
$$
\log\frac{1}{I_{\rm typ}(E)} \;=\; \frac{A}{\varphi^\star(E)^{\kappa}}
\,(1 + o(1)), \qquad \kappa = \tfrac12 \ \text{(H1)},
$$
$\kappa$ kept free as the fallback. This is what makes the crossover *wide*
and the drift *logarithmically slow*: $I_{\rm typ}$ is exponentially small in
$1/\sqrt{\varphi^\star}$ long before any would-be edge.

**Tail scaling of $\varphi^\star$ (hypothesis H2).** In the far tail the
growth of $\mathrm{Im}$ at a site is carried by resonant partners: a
neighbour $j$ contributes when its bond weight $a^2x^2|\mathrm{Re}\,G_j|^2
\gtrsim 1$, and the per-site density of such partners at energy $s$ is the
same annealed tail functional $\propto \langle a^\alpha\rangle\,s^{-\alpha}$
that sets the density tail (3.3, Thm 1(v)). Hence
$$
\varphi^\star(s) \;\simeq\; c\,\langle a^\alpha\rangle\, s^{-\alpha}
\;=\; c\,(s/S)^{-\alpha},
\qquad
S := \langle a^\alpha\rangle^{1/\alpha} \ \ \text{(H2)},
$$
$S$ the row-Levy scale (for the Jacobian, $S = \sigma_w\langle
\phi'(q^{\star 1/\alpha}z)^\alpha\rangle^{1/\alpha}$ in physical units).
Measurable from the seeded-growth diagnostic; the exponent $\alpha$ and the
$S$-scaling are the falsifiable content.

**Resulting drift law.** Substituting H1 + H2 into (8) and keeping the
leading (exponential-in-$1/\sqrt{\varphi^\star}$) factor against the
power-law $\rho$:
$$
\log N \;=\; \frac{A}{c^{1/2}}\,\Big(\frac{s_c}{S}\Big)^{\alpha/2}
+ O(\log s_c)
\qquad\Longrightarrow\qquad
\boxed{\;s_c(N) \;=\; S\,\Big(\frac{c^{1/2}}{A}\,\log N\Big)^{2/\alpha}
\,(1 + o(1)).\;}
\tag{9}
$$
Two structural predictions, independent of the constants:
1. **Scale collapse:** all profile and $\sigma_w$ dependence enters through
   $S$ -- the crossover line in units of the row-Levy scale,
   $s_c/S = E^\star_N$, is a function of $(\alpha, N)$ only. (This is the
   $E^\star(\alpha)$ flatness observed in the Jacobian edge campaign:
   $s_c/S$ constant in $\sigma_w$ to 8-10% per $\alpha$ row.)
2. **Drift law:** $s_c^{\alpha/2}$ is *linear in $\log N$* -- a slow power of
   the logarithm, $s_c \sim (\log N)^{2/\alpha}$ (e.g. $(\log N)^{4/3}$ at
   $\alpha = 1.5$), never saturating: the apparent edge marches to infinity,
   consistent with asymptotic delocalisation.

**Population-dynamics counterpart.** The cavity pool (size $P$, no $\eta$
floor) sustains the weakly delocalised fixed point only while the resonant
sector is sampled: the $m^\star = \tfrac12$ tail of the $\mathrm{Im}$ law
puts mass $\sim (I_{\rm typ}/O(1))^{1/2}$ above the saturation scale, so the
pool loses the fixed point when $P\,I_{\rm typ}^{1/2} \lesssim 1$, i.e.
$\log P \approx \tfrac12\log(1/I_{\rm typ}(s_c)) + O(1)$ -- the same
functional form as (8) with $\log N \to 2\log P$. With $P = 2^{\rm nd}$
(doubling schedule),
$$
\Big(\frac{s_c({\rm nd})}{S}\Big)^{\alpha/2}
\;\approx\; \frac{2\log 2\; c^{1/2}}{A}\,{\rm nd} + {\rm const}:
$$
**$s_c^{\alpha/2}$ linear in nd.** The cavity-vs-empirical density-deviation
edge is exactly this loss-of-resolution point, which is why it drifts with nd
rather than converging -- the drift is the signal, not a numerical failure.

**Campaign check ($\alpha = 1.5$, $\sigma_w = 1$, universal log grid).**
$s_c = 2.57, 2.64, 3.11, 3.61, 3.60(\pm 0.05)$ at
${\rm nd} = 7, 8, 9, 10, 11$ gives
$s_c^{3/4} = 2.03, 2.07, 2.34, 2.62, 2.61$: increments
$0.04, 0.27, 0.27, 0.00$ -- linear in nd at ${\rm nd} = 8$-$10$
(nd $= 7$ sits on the pre-asymptotic floor), then *saturation* at
${\rm nd} = 11$, falsifying the naive pool-limited extrapolation
($s_c^{3/4} \approx 2.88$, $s_c \approx 4.1$). Two follow-ups kill *both*
resolution-ceiling readings of the plateau: (a) re-running the deviation
edge against empirical references at $N_{\rm emp} = 1250/2500/5000$ (SV
counts from the IPR runs) gives $s_c = 3.615/3.627/3.604$ ($\pm 0.05$) --
flat, where an $N_{\rm emp}$-limited edge would move by $\approx +0.5$ per
factor 4 in $N$ given the measured $I_{\rm typ}$ slope (below); (b) the
pool side doubled (nd $10 \to 11$) with no shift, where a pool-limited edge
would move by $\approx +0.5$ likewise. The plateau is therefore neither
pool- nor reference-resolution: beyond $s \approx 3.60$ the empirical
density carries an $N$-independent excess of states over the converged
cavity -- an $O(1)$ localised fraction, i.e. an **asymptotic localisation
edge** $s_\infty \approx 3.60$ at this cell. The IPR exponent crossing
($\tau_{\rm eff}$ of the $625/1250$ vs $2500/5000$ pairs) at $s \approx
2.4$, and $\tau_{\rm eff} \approx 0$-$0.2$ flat in $N$ for $s > 3.9$, are
consistent finite-$N$ shadows of the same edge.

**Validation outcome (`.agents/temp/phi_star_test.py`, 2026-06).**
$\varphi^\star(s)$ (seeded-Im front velocity, $P = 2\times10^4$) and
$I_{\rm typ}(s)$ ($P = 10^4$, $\eta = 10^{-5}, 10^{-6}$) measured on the
structured bipartite cavity at $(1.5, 1)$, unit-scale LePage convention
($s_{\rm phys} = (C_\alpha/2)^{1/\alpha} s_{\rm script} \approx 0.341\,
s_{\rm script}$ by Levy-intensity matching, $C_\alpha = 2\sin(\pi\alpha/2)
\Gamma(\alpha)/\pi$). The measured $\log\lambda(m)$ is linear in $m$ at
every $s$ -- the stationary-front signature: the renormalised population
self-selects the frozen velocity, so the slope *is* $\varphi^\star$
directly (Brunet-Derrida finite-pool bias is downward). Results:
1. **H2 fails in the far tail.** The local slope of $\log\varphi^\star$ vs
   $\log s$ steepens from $-0.75$ ($s_{\rm script} = 4$-$5$) through $-1.5$
   ($6$-$8$) to $-10$ ($10$-$12$): $\varphi^\star$ is not a power law but
   plunges to zero at finite $s^\star \approx 12.5$-$13$ script $\approx
   4.5$-$4.7$ physical ($\theta \approx 0.36$ pinned in (4); a lower bound,
   given the pool bias). The structured
   Jacobian profile has a **true linear-stability edge**: $\lambda(m) \le 1$
   for all $m$ beyond $s^\star$. Mechanism candidate: H2's resonance
   counting lets the profile enter only through $\langle a^\alpha\rangle$,
   but $\log\chi = 2\log\phi'$ inherits the $\alpha$-stable tail of the
   preactivation, $P(|\log\chi| > u) \sim u^{-\alpha}$ -- infinite variance
   for $\alpha < 2$ -- so the DPRM log-gain displacements are heavy-tailed
   and the frozen velocity vanishes at finite $s$: the
   profile-sparsification mechanism of 3.9 / `ht_mlp_jacobian.md` sec. 6,
   operating already at $\sigma_w = 1$.
2. **H1's form holds, exponent open.** $I_{\rm typ}$ falls $2\times10^{-1}
   \to 8\times10^{-7}$ over $s_{\rm script} = 2 \to 10$ and the $P = 10^4$
   pool loses the fixed point at $s_{\rm script} = 12$ (the (8) mechanism
   in vivo); $\log(1/I_{\rm typ})$ diverges as $\varphi^\star \to 0$ with
   $R^2 = 0.97$ at $\kappa = \tfrac12$ but free-$\kappa$ fit $\approx 1.0$
   -- too few near-edge points to pin $\kappa$.
3. **Consequence for (9).** The drift law describes only the intermediate
   regime where the $s^{-\alpha}$ form transiently holds (the nd $= 8$-$10$
   linearity); the apparent edge does not march to infinity but converges
   to $s_\infty$. Prediction 2 above is thereby superseded *for profiles
   with log-singular sparsification*; it stands for bounded-$\log a$
   profiles (incl. unstructured).
4. **Unit factor (resolved 2026-06): the gap is physical.** The
   script $\to$ physical (campaign) factor $\theta$ ($s_{\rm phys} =
   \theta\,s_{\rm script}$) is pinned by two independent routes
   (`.agents/temp/unit_factor_amplitude.py`,
   `.agents/temp/unit_factor_density.py`), which agree. (a) *Bond-amplitude
   scale ratio.* Both conventions' per-step bond sum is a one-sided
   $\alpha/2$-stable differing only in scale (LePage $\Gamma_k^{-2/\alpha}$ vs
   the campaign $(\,(2P)^{-1/\alpha} z_{\rm CMS})^2$); by the exact resolvent
   scaling $w \to \theta^2 w \Leftrightarrow z \to \theta z,\ G \to G/\theta$,
   the singular-value factor is $\theta = (\text{scale ratio})^{1/\alpha}$.
   Tail-robust estimators (median, geometric mean, and $p$-moments at
   $p \le 0.3$, all below $\gamma = \alpha/2 = 0.75$, where fractional moments
   destabilise) give $\theta = 0.350$ in the large-pool stable limit -- matching
   the analytic $(C_\alpha/2)^{1/\alpha} = 0.341$ -- rising to $\theta = 0.367$
   at the as-used $K = 100$ LePage truncation. (b) *Scaling survives the fixed
   point.* The $s \leftrightarrow z$ mapping carries no stray factor by the
   analytic $G \to -G$ algebra alone: numba's $-1/(z + \sum wG)$ equals the
   script's $1/(s - \sum wG)$ at the same $s$. The density test then confirms
   the bond-scale ratio propagates *through the nonlinear cavity fixed point* as
   a pure $s$-axis rescaling -- the LePage- and CMS-convention SV densities (both
   conventions run through one reimplemented bipartite resolvent, the amplitude
   law the only difference) collapse under a single factor $\theta = 0.357$
   across the full bulk-plus-tail, ruling out an $s$-dependent factor. (Neither
   route executes the production numba/phi_star binaries; the tie to those
   pipelines is the $G \to -G$ equivalence plus the campaign-units cross-check
   next.) The density $\theta$ and the amplitude $\theta$ agree at $0.35$-$0.36$.
   (c) *Campaign-units cross-check* (`.agents/temp/unit_factor_xcheck.py`). The
   reimplemented CMS-convention density reproduces the campaign's ground truth --
   the empirical $N = 2500$ SVD density at $(1.5, 1.0)$
   (`fig/jac_emp_log_density/...`) -- in bulk height and falloff onset, with
   median singular value $0.576$ vs the empirical $0.599$ ($\approx 4\%$); the
   only departure is the converged cavity's sharp delocalised-pool collapse at
   $s \approx 3.9$, exactly where the finite-$N$ empirical instead carries its
   localised tail (the deviation edge itself). So the CMS convention *is* physical
   units (factor $\approx 1$), and the script $\to$ physical factor equals the
   LePage $\to$ CMS factor pinned in (a),(b). (The cavity's collapse tail and the
   empirical fat tail differ, so a full-curve shape fit misconverges here; the
   bulk-centred median and the falloff onset are the robust statistics.) The gap-closing value
   $\theta = 0.282$ (which alone would map the $\varphi^\star$-zero onto the
   $3.60$ plateau) is excluded by $> 20\%$. **Verdict:** converting
   $s^\star_{\rm script} \approx 12.5$-$13$ by $\theta \approx 0.36$ puts the
   linearised-stability edge at $s^\star \approx 4.5$-$4.7$ physical, robustly
   above the thermodynamic deviation edge $s_\infty \approx 3.60$ -- the
   linearised $\varphi^\star$-edge sits $\sim 25\%$ above the thermodynamic
   localisation onset, as a linear-stability bound should. The $3.60$-vs-$4.3$
   tension is physics, not units.

**Status of the hypotheses.** Exact inputs: the linearised recursion (4),
the Thouless matching (8) (up to the $O(1)$), the population counterpart's
$\log$ structure. Transplanted/heuristic inputs, each independently
measurable: H1 ($\kappa = \tfrac12$ front selection -- check by measuring
$I_{\rm typ}(s)$ and $\varphi^\star(s)$ on the same grid and regressing
$\log(1/I_{\rm typ})$ on $\varphi^{\star-\kappa}$) and H2 (tail exponent
$\alpha$ and $S$-collapse of $\varphi^\star$ -- check by the seeded-growth
diagnostic across $(\alpha, \sigma_w)$). Gates: (i) $s_c^{\alpha/2}$ linear
in nd in the intermediate (power-law-$\varphi^\star$) regime (passed,
${\rm nd} = 8$-$10$), converging to $s_\infty$ rather than drifting forever
(${\rm nd} = 11$); (ii) the $S$-collapse (passed, the $E^\star(\alpha)$
flatness); (iii) H1/H2 regressions (run: H2 fails in the far tail --
finite-$s$ edge; H1 form passes, $\kappa$ unpinned; see validation outcome
above); (iv) IPR drift slope vs $(\log N)^{2/\alpha}$ (superseded: above a
true edge $\tau_{\rm eff}$ should stay flat in $N$, which it does for
$s > 3.9$); (v) $N_{\rm emp}$-dependence of the deviation plateau (run:
flat at $1250/2500/5000$ -- the edge is asymptotic, not a reference
ceiling); (vi) the script $\to$ physical unit factor (run:
$\theta = 0.35$-$0.36$ from the bond-amplitude scale ratio and the
end-to-end density alignment, in agreement and matching the analytic
$0.341$; the $3.60$-vs-$4.5$ gap between the thermodynamic and
linearised-stability edges is physical, not a convention artifact --
validation outcome (4)).

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
question the structured determinant (Section 3) must answer. *Answered (2026-06)
by Gate B below: it suffices -- a true profile-induced edge exists.*

**Validation gates:**
- **Gate A (done, this section):** index map $\mu = \alpha$, anchored in BG;
  the unstructured operational range is delocalised; `levy_mobility_edge.py`
  consistent. No finite edge to scale-convert.
- **Gate B (done, 2026-06; row profile $a_i = \sigma_w|\phi'|$):** *a bounded row
  profile induces a true edge.* The decisive test -- tail-$s$ singular-vector
  $D_2$ $N$-scaling (IPR $\sim N^{-D_2}$; slope $\mathrm dD_2/\mathrm d\log N \to 0$
  = true edge, the BG-delocalised baseline keeps $\to +$ toward $D_2=1$) -- run as
  an $N=1000\dots6000$ ladder at $(\alpha,\sigma_w)=(1.5,3.0)$
  (`.agents/temp/n_ladder_d2.py`, 3.6). Outcome: the unstructured baseline $D_2$
  climbs at every $s$; the structured tail $D_2$ goes **flat** ($s{=}6.5$:
  $0.09\to0.09$; $s{=}8$: $0.05\to0.05$ over the same $N$-window where the baseline
  climbs $0.35\to0.41$, $0.21\to0.28$), crossing from delocalising to frozen
  between $s\approx5$ and $6.5$. So the saturated-unit profile *does* push a
  genuine asymptotic localisation onset into the operational tail -- not a mere
  finite-$N$ crossover. Open: whether the frozen tail $D_2$ is $0$ (Anderson) or a
  nonzero critical value (3.6).

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
mobility edge then predicts the profile-induced shift of $s_c$ relative to
the unstructured baseline of Section 4 -- i.e. whether saturated-unit structure
pushes the localisation onset into the operational bulk. (Physical singular-value
units carry the global entry scale: $s_c^{\rm phys} = \sigma_w\,s_c$, since the
Jacobian entry is $\sigma_w$-scaled and localisation is invariant under global
rescaling.) The full derivation, solver wiring, and comparison to a direct
MLP-Jacobian IPR sweep live in `ht_mlp_jacobian.{md,py,ipynb}`. This file states
only the general result.

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
