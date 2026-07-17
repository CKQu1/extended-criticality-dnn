# Hermitisation and the bipartite cavity method for singular-value spectra

The non-Hermitian singular-value problem for a rectangular matrix $A$ is
turned into a Hermitian eigenvalue problem by **Hermitisation**. The
Hermitisation supports a tree-closed cavity recursion, whose bipartite block
structure yields a coupled fixed point for the two resolvent populations.
The singular-value density is read off by the standard complex-$z$ Stieltjes
inversion of the resolvent.

## Hermitisation: eigenvalue <-> singular-value bridge -- eq. (10)

Consider the non-Hermitian matrix $A$ of size $N\times M$, where $N\leq M$, and denote its Hermitisation by

$$H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}.$$

Its square is a matrix with diagonal blocks $AA^\dagger$ and $A^\dagger A$ and off-diagonal blocks zero, while $AA^\dagger$ has the $M$ eigenvalues of $A^\dagger A$ (i.e. the squared singular values of $A$) along with $N-M$ zero eigenvalues. Thus the eigenvalues of $H$ are formed from two copies of the $M$ singular values of $A$ along with $N-M$ zeroes,

$$\mu_H = \frac{2M}{N+M} \nu_A + \frac{N-M}{N+M} \delta_0 ~. \qquad\text{(10)}$$

## Cavity recursion on the Hermitisation

For complex spectral parameter $z$, the diagonal resolvent entries $G_i(z) := (H-z)^{-1}_{ii}$ obey the cavity equation

$$(H-z)^{-1}_{ii} = \left( H_{ii} - z - \sum_{j,k\neq i} H_{ij} (H^{(i)} - z)^{-1}_{jk} H_{ki} \right)^{-1}$$

where $H^{(i)}$ is the submatrix with index $i$ removed.

Because heavy-tailed matrices are locally treelike [cite], the sum is restricted to $j$ and $k$ over the neighbouring nodes of $i$ on the tree on which $H$ acts as a weighted adjacency matrix. When $i$ is removed, $H^{(i)}$ acts on a forest of isolated subtrees with the neighbours of $i$ as their roots, and so by permuting the block inverse formula one can show that the off-diagonal entries of the resolvent $(H^{(i)} - z)^{-1}_{jk} = 0$ when $j\neq k$. This restricts the sum to $\sum_{j\in\partial_i}H_{ij}(H^{(i)}-z)^{-1}_{jj}H_{ki}$. Because the resolvent of a matrix is not *a priori* equal to that of its submatrix, we remove another index from the submatrix and express the resolvent for $j\in\partial_i$, the neighbours of $i$:

$$(H^{(i)}-z)^{-1}_{jj} = \left( H_{jj} - z - \sum_{k\in\partial_j\setminus i} H_{jk} (H^{(i, j)} - z)^{-1}_{kk} H_{kj} \right)^{-1} ~.$$

Because the subforests rooted at $k\in\partial_j\setminus i$ are disconnected from $i$ when only $j$ is removed, the resolvent at $k$ is independent of the existence of $i$, so that $(H^{(i, j)} - z)^{-1}_{kk} = (H^{(j)} - z)^{-1}_{kk}$. Since the subtrees rooted at $j$ and $k$ are statistically identical over $\partial_i$ and $\partial_j\setminus i$ respectively, the sum can be performed over $k\in\partial_i$ instead, closing the recursion:

$$(H^{(i)}-z)^{-1}_{jj} = \left( H_{jj} - z - \sum_{k\in\partial_i} H_{jk} (H^{(i)} - z)^{-1}_{kk} H_{kj} \right)^{-1} ~.$$

Regarding $H$ itself as the submatrix of a matrix with one more row and column yields a closed form of the resolvent using the cavity method,

$$(H-z)^{-1}_{ii} = \left( H_{ii} - z - \sum_{j\in\partial_i} H_{ij} (H - z)^{-1}_{jj} H_{ji} \right)^{-1} ~.$$

The derivation makes no assumption on whether $z$ is real or complex; the same recursion drives the density readout in either limit.

## Bipartite split -- eqs. (12)-(13)

The cavity equation has a bipartite structure due to the Hermitisation, which can be expressed as a coupled set of equations by defining $G_i^{(1)}(z) = (H-z)^{-1}_{ii}$ for $1\leq i\leq M$ and $G_j^{(2)}(z) = (H-z)^{-1}_{(M+j)(M+j)}$ for $1\leq j\leq N$. Then

$$\begin{aligned} G_i^{(1)}(z) &= -\left(z + \sum_{j=1}^N |A_{ij}|^2 G_j^{(2)}(z) \right)^{-1} & \qquad\text{(12)} \\ G_j^{(2)}(z) &= -\left(z + \sum_{i=1}^M |A_{ij}|^2 G_i^{(1)}(z) \right)^{-1} & \qquad\text{(13)} \end{aligned}$$

## Density readout

**Complex-$z$ Stieltjes inversion.** Set $z = s + i\varepsilon$ and use

$$\rho_H(s) = -\frac{1}{\pi}\lim_{\varepsilon\downarrow 0}\,\mathrm{Im}\,G(s + i\varepsilon),$$

solving (12)-(13) with complex $G_i^{(1)}, G_j^{(2)}$ and reading off

$$\rho(s) = \frac{1}{\pi\,(N+M)}\!\left(\sum_i \mathrm{Im}\,G_i^{(1)}(s+i\varepsilon) + \sum_j \mathrm{Im}\,G_j^{(2)}(s+i\varepsilon)\right).$$

This is the route used by `RMT.py:cavity_svd_resolvent` (with the regulariser $\varepsilon$ injected in `jac_cavity_svd_log_pdf`) and by the analytical theory in `RMT/structured_wishart_levy.py`. (An equivalent real-axis self-energy readout exists in the literature but is not used anywhere in this repo.)
