
## Singular values of the Jacobian of heavy-tailed DNNs

Consider the non-Hermitian matrix $A$ of size $N\times M$, where $N\leq M$, and denote its Hermitisation by

$$H = \begin{pmatrix} 0 & A \\ A^\dagger & 0 \end{pmatrix}.$$

Its square is a matrix with diagonal blocks $AA^\dagger$ and $A^\dagger A$ and off-diagonal blocks zero, while $AA^\dagger$ has the $M$ eigenvalues of $A^\dagger A$ (i.e. the squared singular values of $A$) along with $N-M$ zero eigenvalues. Thus the eigenvalues of $H$ are formed from two copies of the $M$ singular values of $A$ along with $N-M$ zeroes,

$$\mu_H = \frac{2M}{N+M} \nu_A + \frac{N-M}{N+M} \delta_0 ~.$$

To compute the eigenvalue density of $H$, we follow Cizeau & Bouchaud (1994) and work directly at real spectral parameter $s$. For Hermitian $H$, the diagonal resolvent entries $G_i(s) := (H-s)^{-1}_{ii}$ are real-valued at real $s$. Writing $G_i(s) = -(s+\Sigma_i(s))^{-1}$, each self-energy $\Sigma_i(s)$ determines a pole of $G_i$ at $s = -\Sigma_i$, i.e.\ an eigenvalue of $H$. In the large-$N$ limit the self-energies $\{\Sigma_i(s)\}$ are i.i.d.\ draws from a distribution $p_{\Sigma(s)}$, so the eigenvalue density equals the probability density of $\Sigma_i$ evaluated at $-s$:

$$\rho_H(s) = p_{\Sigma(s)}(-s).$$

The self-energy satisfies the cavity equation for the real resolvent:

$$(H-s)^{-1}_{ii} = \left( H_{ii} - s - \sum_{j,k\neq i} H_{ij} (H^{(i)} - s)^{-1}_{jk} H_{ki} \right)^{-1}$$

where $H^{(i)}$ is the submatrix with index $i$ removed.

We evaluate the resolvent using a cavity approach. Because heavy-tailed matrices are locally treelike [cite], the sum is restricted to $j$ and $k$ over the neighbouring nodes of $i$ on the tree on which $H$ acts as a weighted adjacency matrix. When $i$ is removed, $H^{(i)}$ acts on a forest of isolated subtrees with the neighbours of $i$ as their roots, and so by permuting the block inverse formula one can show that the off-diagonal entries of the resolvent $(H^{(i)} - z)^{-1}_{jk} = 0$ when $j\neq k$. This restricts the sum to $\sum_{j\in\partial_i}H_{ij}(H^{(i)}-z)^{-1}_{jj}H_{ki}$. Because the resolvent of a matrix is not *a priori* equal to that of its submatrix, we remove another index from the submatrix and express the resolvent for $j\in\partial_i$, the neighbours of $i$:

$$(H^{(i)}-z)^{-1}_{jj} = \left( H_{jj} - z - \sum_{k\in\partial_j\setminus i} H_{jk} (H^{(i, j)} - z)^{-1}_{kk} H_{kj} \right)^{-1} ~.$$

Because the subforests rooted at $k\in\partial_j\setminus i$ are disconnected from $i$ when only $j$ is removed, the resolvent at $k$ is independent of the existence of $i$, so that $(H^{(i, j)} - z)^{-1}_{kk} = (H^{(j)} - z)^{-1}_{kk}$. Since the subtrees rooted at $j$ and $k$ are statistically identical over $\partial_i$ and $\partial_j\setminus i$ respectively, the sum can be performed over $k\in\partial_i$ instead, closing the recursion:

$$(H^{(i)}-z)^{-1}_{jj} = \left( H_{jj} - z - \sum_{k\in\partial_i} H_{jk} (H^{(i)} - z)^{-1}_{kk} H_{kj} \right)^{-1} ~.$$

Regarding $H$ itself as the submatrix of a matrix with one more row and column yields a closed form of the resolvent using the cavity method,

$$(H-z)^{-1}_{ii} = \left( H_{ii} - z - \sum_{j\in\partial_i} H_{ij} (H - z)^{-1}_{jj} H_{ji} \right)^{-1} ~.$$

The cavity equation has a bipartite structure due to the Hermitisation, which can be expressed as a coupled set of equations by defining $G_i^{(1)}(z) = (H-z)^{-1}_{ii}$ for $1\leq i\leq M$ and $G_j^{(2)}(z) = (H-z)^{-1}_{(M+j)(M+j)}$ for $1\leq j\leq N$. Then

$$\begin{aligned} G_i^{(1)}(z) &= -\left(z + \sum_{j=1}^N |A_{ij}|^2 G_j^{(2)}(z) \right)^{-1} \\ G_j^{(2)}(z) &= -\left(z + \sum_{i=1}^M |A_{ij}|^2 G_i^{(1)}(z) \right)^{-1} \end{aligned}$$

Finally, the relation between the eigenvalues of $H$ and the singular values of $A$ gives the singular value density of $A$. In the bipartite structure the type-1 and type-2 self-energies are

$$\Sigma_i^{(1)}(s) = \sum_{j=1}^N |A_{ij}|^2\,G_j^{(2)}(s), \qquad \Sigma_j^{(2)}(s) = \sum_{i=1}^M |A_{ij}|^2\,G_i^{(1)}(s),$$

and equal numbers of eigenvalues arise from poles of each type. The singular value density of $A$ is therefore

$$\rho(s) = \frac{1}{2}\!\left(p_{\Sigma^{(1)}(s)}(-s) + p_{\Sigma^{(2)}(s)}(-s)\right).$$
