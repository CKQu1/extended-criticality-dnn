
## Analytical singular value density of heavy-tailed MLP Jacobians

We derive the singular value density of the layerwise Jacobian of a random MLP with heavy-tailed weights using the cavity approach of `sing_vals.md`, closed analytically via the generalised central limit theorem following [Wardak & Gong 2022].

### Notation

A stable distribution $L(\alpha, \beta, \gamma, D)$ has characteristic function

$$\langle e^{ikL} \rangle = e^{ik\gamma - D|k|^\alpha(1 - i\beta\,\mathrm{sgn}(k)\tan(\alpha\pi/2))}$$

with index $\alpha$, skewness $\beta \in [-1,1]$, centre $\gamma$, and scale $D$. We write $C_\alpha := \Gamma(1+\alpha)\sin(\pi\alpha/2)/\pi$.

---

### 1. Setup

Consider a single-layer weight matrix $W$ with iid entries

$$W_{ij} \sim L\!\left(\alpha,\, 0,\, 0,\, \frac{\sigma_W^\alpha}{2N}\right), \qquad 1 \leq i,j \leq N,$$

so that $P(|W_{ij}| > t) \sim \frac{C_\alpha \sigma_W^\alpha}{2N} t^{-\alpha}$ for large $t$. This matches the code normalisation `sigma_W * (2*N)**(-1/alpha) * xi` where $\xi \sim L(\alpha,0,0,1/2)$.

The **pre-activation Jacobian** at layer $l$ is the column-structured matrix

$$A_{ij} = W_{ij}\,\chi_j, \qquad \chi_j := \phi'(h_j^*),$$

where $h_j^*$ is the stationary local field at the fixed point, whose distribution is determined self-consistently below. Setting $\chi_j = 1$ for all $j$ recovers the pure weight matrix $A = W$.

The stationary field distribution follows from the Lévy mean-field equation [Wardak & Gong 2022, Supp. §III.C]:

$$h^* \sim L\!\left(\alpha,\, 0,\, 0,\, \frac{\sigma_W^\alpha \langle|\phi(h^*)|^\alpha\rangle}{2}\right), \qquad \langle|\phi(h^*)|^\alpha\rangle = \int |\phi(h)|^\alpha\, p_{h^*}(h)\,dh.$$

This fixed-point equation is solved numerically by `q_star_MC` in `RMT.py`.

---

### 2. CLT closure of the cavity equations

From `sing_vals.md`, the diagonal elements of the resolvent of the Hermitian bipartisation $H = \begin{pmatrix}0 & A \\ A^\dagger & 0\end{pmatrix}$ satisfy

$$G_i^{(1)}(z) = -\!\left(z + \sum_{j=1}^N |A_{ij}|^2\, G_j^{(2)}(z)\right)^{-1}, \qquad G_j^{(2)}(z) = -\!\left(z + \sum_{i=1}^N |A_{ij}|^2\, G_i^{(1)}(z)\right)^{-1}.$$

Substituting $A_{ij} = W_{ij}\chi_j$:

$$G_i^{(1)} = -\!\left(z + \sum_j |\chi_j|^2\,|W_{ij}|^2\, G_j^{(2)}\right)^{-1}, \qquad G_j^{(2)} = -\!\left(z + |\chi_j|^2 \sum_i |W_{ij}|^2\, G_i^{(1)}\right)^{-1}.$$

**Real-axis cavity resolvent.** For a Hermitian matrix $H$ and real $z = s$, the diagonal resolvent entries are real: $G_i^{(1)} = u_i \in \mathbb{R}$ and $G_j^{(2)} = x_j \in \mathbb{R}$. The self-energies are accordingly real,

$$A_i \;:=\; \sum_j |\chi_j|^2|W_{ij}|^2\, x_j \;\in\; \mathbb{R}, \qquad C_j \;:=\; \sum_i |W_{ij}|^2\, u_i \;\in\; \mathbb{R},$$

and the cavity equations at real $s$ reduce to

$$u_i = -\frac{1}{s + A_i}, \qquad x_j = -\frac{1}{s + |\chi_j|^2 C_j}.$$

For $s$ inside the singular value spectrum, $u_i$ and $x_j$ can be of either sign (depending on whether $s + A_i$ and $s + |\chi_j|^2 C_j$ are positive or negative), so $A_i$ is a sum of positive heavy-tailed weights multiplied by real coefficients of mixed sign.

**Applying the generalised CLT.** The entries $|W_{ij}|^2$ are iid positive $(\alpha/2)$-stable with tail

$$P\!\left(|W_{ij}|^2 > t\right) \sim \frac{C_\alpha\sigma_W^\alpha}{2N}\,t^{-\alpha/2}.$$

Since $|\chi_j|^2 x_j \in \mathbb{R}$ is independent of $|W_{ij}|^2$ by the locally treelike structure, the generalised CLT gives $A_i$ as a **real** $(\alpha/2)$-stable:

$$A_i \;\sim\; L\!\left(\frac{\alpha}{2},\;\frac{\tilde{q}}{q},\;0,\;\frac{C_\alpha\sigma_W^\alpha}{2}\,q\right),$$

parametrised by the scale and skewness moments

$$q \;:=\; \bigl\langle|\chi|^\alpha|G^{(2)}|^{\alpha/2}\bigr\rangle, \qquad \tilde{q} \;:=\; \bigl\langle|\chi|^\alpha|G^{(2)}|^{\alpha/2}\operatorname{sgn}(G^{(2)})\bigr\rangle,$$

with $\tilde{q}/q \in [-1,1]$ the skewness. Analogously, $C_j$ is a real $(\alpha/2)$-stable:

$$C_j \;\sim\; L\!\left(\frac{\alpha}{2},\;\frac{\tilde{p}}{p},\;0,\;\frac{C_\alpha\sigma_W^\alpha}{2}\,p\right),$$

with

$$p \;:=\; \bigl\langle|G^{(1)}|^{\alpha/2}\bigr\rangle, \qquad \tilde{p} \;:=\; \bigl\langle|G^{(1)}|^{\alpha/2}\operatorname{sgn}(G^{(1)})\bigr\rangle.$$

---

### 3. Self-consistent equations

Substituting $G^{(1)} = -(s+A)^{-1}$ and $G^{(2)} = -(s+|\chi|^2C)^{-1}$ gives

$$p \;=\; \left\langle\frac{1}{|s+A|^{\alpha/2}}\right\rangle_{\!A}, \qquad \tilde{p} \;=\; -\left\langle\frac{\operatorname{sgn}(s+A)}{|s+A|^{\alpha/2}}\right\rangle_{\!A},$$

$$q \;=\; \left\langle\frac{|\chi|^\alpha}{|s+|\chi|^2 C|^{\alpha/2}}\right\rangle_{\!\chi,C}, \qquad \tilde{q} \;=\; -\left\langle\frac{|\chi|^\alpha\operatorname{sgn}(s+|\chi|^2 C)}{|s+|\chi|^2 C|^{\alpha/2}}\right\rangle_{\!\chi,C},$$

where $A \sim L\!\left(\tfrac{\alpha}{2},\,\tfrac{\tilde{q}}{q},\,0,\,\tfrac{C_\alpha\sigma_W^\alpha}{2}q\right)$ and $C \sim L\!\left(\tfrac{\alpha}{2},\,\tfrac{\tilde{p}}{p},\,0,\,\tfrac{C_\alpha\sigma_W^\alpha}{2}p\right)$, and the averages $\langle\cdot\rangle_\chi$ are over the stationary field distribution. These are four coupled scalar fixed-point equations for $(p,\,q,\,\tilde{p},\,\tilde{q})\in\mathbb{R}_+^2\times[-p,p]\times[-q,q]$.

---

### 4. Density of states

From the C&B formula of `sing_vals.md`, the singular value density at $s$ equals the probability density of the self-energy evaluated at $-s$. The type-1 self-energy is $\Sigma^{(1)} = A_i \sim L\!\left(\tfrac{\alpha}{2},\,\tfrac{\tilde{q}}{q},\,0,\,\tfrac{C_\alpha\sigma_W^\alpha}{2}q\right)$ and the type-2 self-energy is $\Sigma^{(2)} = |\chi_j|^2 C_j$, where $C_j \sim L\!\left(\tfrac{\alpha}{2},\,\tfrac{\tilde{p}}{p},\,0,\,\tfrac{C_\alpha\sigma_W^\alpha}{2}p\right)$. For $N=M$ (square $A$) the singular value density at real $s$ is therefore

$$\boxed{\rho(s) \;=\; \frac{1}{2}\Bigl(p_A(-s) \;+\; \bigl\langle p_{|\chi|^2 C}(-s)\bigr\rangle_\chi\Bigr),}$$

where $p_A$ is the density of $A\sim L\!\left(\tfrac{\alpha}{2},\,\tfrac{\tilde{q}}{q},\,0,\,\tfrac{C_\alpha\sigma_W^\alpha}{2}q\right)$ and $p_{|\chi|^2C}$ is the density of the scaled stable $|\chi|^2 C$, both evaluated at $-s$ at the fixed point $(p,q,\tilde{p},\tilde{q})$ of §3.

---

### 5. Power-law tail

We show that $\rho(s) \sim s^{-1-\alpha}$ as $s \to +\infty$.

**Step 1: large-$s$ fixed-point scaling.** For $s$ large the self-energies $A$ and $|\chi|^2 C$ are small compared to $s$, so $G^{(1)} \approx -1/s$ and $G^{(2)} \approx -1/s$. Substituting into the definitions of $p$ and $q$:

$$p \;=\; \left\langle |s+A|^{-\alpha/2} \right\rangle_A \;\sim\; s^{-\alpha/2}, \qquad
  q \;=\; \left\langle \frac{|\chi|^\alpha}{|s+|\chi|^2 C|^{\alpha/2}} \right\rangle \;\sim\; \langle|\chi|^\alpha\rangle\, s^{-\alpha/2}.$$

Hence the scale parameters of the self-energy distributions behave as

$$D_A \;:=\; \frac{C_\alpha\sigma_W^\alpha}{2}\,q \;\sim\; \frac{C_\alpha\sigma_W^\alpha\langle|\chi|^\alpha\rangle}{2}\,s^{-\alpha/2},
\qquad
D_C \;:=\; \frac{C_\alpha\sigma_W^\alpha}{2}\,p \;\sim\; \frac{C_\alpha\sigma_W^\alpha}{2}\,s^{-\alpha/2}.$$

Both $D_A$ and $D_C$ vanish as $s \to \infty$, confirming that $A$ and $|\chi|^2 C$ are $O(s^{-1})$ fluctuations around zero, as assumed.

**Step 2: stable tail formula.** For any $L(\alpha_s, \beta, 0, D)$ with $\alpha_s \in (0,1)$ and skewness $\beta \in [-1,1]$, the left-tail density satisfies

$$p_{L(\alpha_s,\beta,0,D)}(-s) \;\sim\; \frac{\alpha_s\, C_{\alpha_s}(1-\beta)\, D}{s^{1+\alpha_s}} \quad s\to+\infty,$$

where $C_{\alpha_s} = \Gamma(1+\alpha_s)\sin(\pi\alpha_s/2)/\pi$.

**Step 3: type-1 contribution.** Applying the tail formula with $\alpha_s = \alpha/2$ and $D = D_A \sim C_\alpha\sigma_W^\alpha\langle|\chi|^\alpha\rangle s^{-\alpha/2}/2$:

$$p_A(-s) \;\sim\; \frac{(\alpha/2)\,C_{\alpha/2}(1-\beta_A)\,C_\alpha\sigma_W^\alpha\langle|\chi|^\alpha\rangle}{2}\,\frac{1}{s^{1+\alpha/2}\cdot s^{\alpha/2}} \;=\; O\!\left(s^{-1-\alpha}\right).$$

**Step 4: type-2 contribution.** For a fixed $\chi \neq 0$, the density of $|\chi|^2 C$ at $-s$ is obtained by rescaling:

$$p_{|\chi|^2 C}(-s) \;=\; \frac{1}{|\chi|^2}\,p_C\!\left(-\frac{s}{|\chi|^2}\right)
\;\sim\; \frac{(\alpha/2)\,C_{\alpha/2}(1-\beta_C)\,D_C\,|\chi|^{2\cdot\alpha/2}}{s^{1+\alpha/2}}
\;=\; \frac{(\alpha/2)\,C_{\alpha/2}(1-\beta_C)\,D_C\,|\chi|^\alpha}{s^{1+\alpha/2}}.$$

Averaging over $\chi$ and substituting $D_C \sim C_\alpha\sigma_W^\alpha s^{-\alpha/2}/2$:

$$\bigl\langle p_{|\chi|^2 C}(-s)\bigr\rangle_\chi \;\sim\; \frac{(\alpha/2)\,C_{\alpha/2}(1-\beta_C)\,C_\alpha\sigma_W^\alpha\langle|\chi|^\alpha\rangle}{2}\,\frac{1}{s^{1+\alpha}} \;=\; O\!\left(s^{-1-\alpha}\right).$$

**Conclusion.** Both terms in the DOS formula are $O(s^{-1-\alpha})$, giving

$$\boxed{\rho(s) \;\sim\; K\,s^{-1-\alpha} \quad s\to+\infty,}$$

where the prefactor $K$ depends on $\alpha$, $\sigma_W$, and $\langle|\chi|^\alpha\rangle$ through the constants above. The power-law exponent $-1-\alpha$ is the same as the tail of the original weight entries $W_{ij}$, confirming that the heavy tail of the singular value spectrum is inherited directly from the weight distribution.
