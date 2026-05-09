# Column-scaled Wishart-Lévy: derivation scaffold

This note prepares a derivation of the limiting singular-value density for a
rectangular heavy-tailed random matrix with a deterministic **column profile**.
The intended route is **not** the real-axis cavity heuristic in `sing_vals.md`,
but the general profile theorem of Belinschi--Dembo--Guionnet specialized to the
Hermitization of a column-scaled rectangular matrix.

The algebraic reduction below has been checked against the structure of
Belinschi's general profile theorem. The main remaining issue is not the block
algebra, but the precise modeling/continuum justification of the chosen profile
specialization.

## 1. Model and normalization

Let $X_{N,M}$ be an $N \times M$ matrix with iid entries in the domain of
attraction of an $\alpha$-stable law, with

$$
\frac{M}{N} \to \gamma \in (0,1].
$$

Let the column-scaled matrix be

$$
A_{N,M}(i,j) = c_j x_{ij},
$$

where the deterministic column weights are sampled from a profile
$c : [0,1] \to [0,\infty)$ in the large-$N$ limit.

For the Belinschi derivation it is most natural to work first in the paper's
quantile normalization,

$$
\widetilde A_{N,M} = a_{N+M}^{-1} A_{N,M},
$$

and only afterwards convert back to the direct stable-sampling normalization if
needed.

Define the heavy-tailed covariance / Wishart matrix

$$
W_{N,M} = \widetilde A_{N,M} \widetilde A_{N,M}^T.
$$

Its eigenvalues are the squared singular values of $\widetilde A_{N,M}$.

## 2. Hermitization

Introduce the Hermitized matrix

$$
B_{N+M} =
\begin{pmatrix}
0 & \widetilde A_{N,M} \\
\widetilde A_{N,M}^T & 0
\end{pmatrix}.
$$

If $s_1,\dots,s_M$ are the singular values of $\widetilde A_{N,M}$, then the
nonzero eigenvalues of $B_{N+M}$ are

$$
\pm s_1,\dots,\pm s_M,
$$

and there are $N-M$ extra zero eigenvalues.

So the rectangular singular-value problem is reduced to the spectral measure of
a symmetric heavy-tailed matrix with a special off-diagonal profile.

## 3. Belinschi's general profile theorem

For a symmetric profile $\sigma(x,y)$ on $[0,1]^2$, Belinschi's general theorem
gives a limiting spectral measure $\mu^\sigma$ with Cauchy-Stieltjes transform

$$
G_\sigma(z) = \int \frac{1}{z-x} d\mu^\sigma(x)
=
\frac{1}{z}\int_0^1 h_\alpha(Y_x^\sigma(z))\,dx,
$$

where $Y^\sigma_x(z)$ solves

$$
z^\alpha Y_x^\sigma(z)
=
C_\alpha \int_0^1 |\sigma(x,v)|^\alpha g_\alpha(Y_v^\sigma(z))\,dv.
$$

Here

$$
g_\alpha(y) = \int_0^\infty t^{\alpha/2-1} e^{-t} e^{-t^{\alpha/2}y}\,dt,
$$

$$
h_\alpha(y) = \int_0^\infty e^{-t} e^{-t^{\alpha/2}y}\,dt
=
1-\frac{\alpha}{2}y g_\alpha(y),
$$

and

$$
C_\alpha = i^\alpha \frac{\Gamma(1-\alpha/2)}{\Gamma(\alpha/2)}.
$$

## 4. Profile for column scaling

Split the unit interval into a row block and a column block:

$$
I_r = \left(0,\frac{1}{1+\gamma}\right],
\qquad
I_c = \left(\frac{1}{1+\gamma},1\right].
$$

Let $u \in [0,1]$ parameterize the column block by

$$
v = \frac{1}{1+\gamma} + \frac{\gamma}{1+\gamma}u.
$$

Then the natural Hermitized profile is

$$
\sigma_c(x,y) =
\begin{cases}
c(u(y)), & x \in I_r,\ y \in I_c, \\
c(u(x)), & x \in I_c,\ y \in I_r, \\
0, & \text{otherwise},
\end{cases}
$$

where

$$
u(v) = \frac{1+\gamma}{\gamma}\left(v-\frac{1}{1+\gamma}\right)
$$

maps the column block back to $[0,1]$.

This is the column-scaled analogue of the constant off-diagonal profile used by
Belinschi for ordinary covariance matrices.

## 5. Reduction of the profile equations

For this profile, the row side is homogeneous while the column side still
retains the profile $c(u)$. Accordingly, the profile field reduces to

$$
Y_x^\sigma(z) =
\begin{cases}
Y_r(z), & x \in I_r, \\
Y_c(u(x);z), & x \in I_c.
\end{cases}
$$

Substituting this ansatz into the general profile theorem gives

$$
z^\alpha Y_r(z)
=
\frac{\gamma C_\alpha}{1+\gamma}
\int_0^1 c(u)^\alpha g_\alpha(Y_c(u;z))\,du,
$$

$$
z^\alpha Y_c(u;z)
=
\frac{C_\alpha}{1+\gamma}
c(u)^\alpha g_\alpha(Y_r(z)).
$$

The second equation is explicit:

$$
Y_c(u;z)
=
\frac{C_\alpha}{1+\gamma}
z^{-\alpha} c(u)^\alpha g_\alpha(Y_r(z)).
$$

Plugging this back into the first equation yields the desired **scalar closure**
for $Y_r(z)$:

$$
z^\alpha Y_r(z)
=
\frac{\gamma C_\alpha}{1+\gamma}
\int_0^1 c(u)^\alpha
g_\alpha\!\left(
\frac{C_\alpha}{1+\gamma}
z^{-\alpha} c(u)^\alpha g_\alpha(Y_r(z))
\right) du.
$$

This is the central equation to derive cleanly in the final writeup.

## 6. Consistency check: the unscaled covariance case

If

$$
c(u) \equiv 1,
$$

then $Y_c(u;z)$ becomes independent of $u$, say $Y_c(u;z)=Y_2(z)$, and the
column-scaled system collapses to Belinschi's two-function covariance system:

$$
z^\alpha Y_1(z)=\frac{\gamma}{1+\gamma} C_\alpha g_\alpha(Y_2(z)),
\qquad
z^\alpha Y_2(z)=\frac{1}{1+\gamma} C_\alpha g_\alpha(Y_1(z)).
$$

So the proposed specialization is at least structurally consistent with the
known Wishart-Lévy formula.

## 7. Stieltjes transform for the Hermitized law

For the profile $\sigma_c$,

$$
z G_{\sigma_c}(z)
=
\frac{1}{1+\gamma} h_\alpha(Y_r(z))
+
\frac{\gamma}{1+\gamma} \int_0^1 h_\alpha(Y_c(u;z))\,du.
$$

Multiplying the row equation by $g_\alpha(Y_r(z))$ and the column equation by
$Y_c(u;z) g_\alpha(Y_c(u;z))$, then integrating the latter over $u$, gives

$$
Y_r(z) g_\alpha(Y_r(z))
=
\gamma \int_0^1 Y_c(u;z) g_\alpha(Y_c(u;z))\,du.
$$

Using

$$
h_\alpha(y)=1-\frac{\alpha}{2}y g_\alpha(y),
$$

this becomes

$$
h_\alpha(Y_r(z))
=
1-\gamma
+
\gamma \int_0^1 h_\alpha(Y_c(u;z))\,du.
$$

This identity lets the final formula collapse to the row-side quantity alone.

## 8. From Hermitization to squared singular values

Let $\nu_c$ denote the limiting measure of the squared singular values, i.e. the
limiting spectral measure of $W_{N,M}$.

Exactly as in the ordinary Wishart-Lévy case, the symmetry of the Hermitized law
and the rectangular zero modes imply

$$
G_{\nu_c}(z)
=
\frac{1+\gamma}{2\sqrt{z}} G_{\sigma_c}(\sqrt{z})
+
\frac{1-\gamma}{2z}.
$$

Using the identity from the previous section, this reduces to

$$
G_{\nu_c}(z) = \frac{1}{z} h_\alpha(Y_r(\sqrt{z})).
$$

Hence the squared singular-value density should be

$$
\rho_c(t)
=
-\frac{1}{\pi t}\Im\!\big(h_\alpha(Y_r(\sqrt{t}))\big),
\qquad t>0.
$$

And the singular-value density is then

$$
f_c(s) = 2s\,\rho_c(s^2).
$$

So the entire column-scaled problem reduces to solving the single scalar
fixed-point equation for $Y_r(z)$.

## 9. Tail consequence

Belinschi's general profile theorem states that for a symmetric profile
$\sigma(x,y)$,

$$
t^{1+\alpha}\rho^\sigma(t) \to \frac{\alpha}{2}\int_0^1\int_0^1
|\sigma(x,v)|^\alpha\,dx\,dv.
$$

For the column-scaled Hermitized profile,

$$
\int_0^1\int_0^1 |\sigma_c(x,v)|^\alpha\,dx\,dv
=
\frac{2\gamma}{(1+\gamma)^2}\int_0^1 c(u)^\alpha\,du.
$$

Hence the Hermitized density should satisfy

$$
\rho_{\sigma_c}(s)
\sim
\frac{\alpha\gamma}{(1+\gamma)^2}
\left(\int_0^1 c(u)^\alpha\,du\right)
|s|^{-1-\alpha}.
$$

Pushing forward under $s \mapsto s^2$ and then undoing the rectangular zero-mode
mixture exactly as in the ordinary Wishart-Lévy case gives the squared
singular-value tail

$$
\rho_c(t)
\sim
\frac{\alpha\gamma}{2(1+\gamma)}
\left(\int_0^1 c(u)^\alpha\,du\right)
t^{-1-\alpha/2},
\qquad t\to\infty.
$$

Equivalently, the singular-value density should satisfy

$$
f_c(s)
\sim
\frac{\alpha\gamma}{1+\gamma}
\left(\int_0^1 c(u)^\alpha\,du\right)
s^{-1-\alpha},
\qquad s\to\infty.
$$

For $c(u)\equiv 1$, this reduces to Belinschi's ordinary Wishart-Lévy tail law.

## 10. Remaining checks

The final derivation should still verify the following points carefully:

1. the precise discrete-to-continuum definition of the column profile $c(u)$;
2. the precise discrete-to-profile embedding of the Hermitized column-scaled
   matrix into Belinschi's $\sigma(x,y)$ framework;
3. the regularity/branch choices needed to evaluate the boundary value
   $Y_r(\sqrt{t})$ on the positive real axis;
4. the conversion back to the direct stable-sampling normalization used in
   numerical work.

## 11. Intended outcome

If the derivation closes as expected, the column-scaled Wishart-Lévy theory
curve should have the same overall structure as the Belinschi covariance law,
but with the two coupled scalar unknowns replaced by

1. one scalar row-side unknown $Y_r(z)$, and
2. one explicit profile-dependent column field $Y_c(u;z)$.

That would give a practical route to theory curves for singular values of
column-scaled heavy-tailed random matrices without going back to the real-axis
cavity heuristic.
