# Full network Jacobian: analytical route

For a single layer,

$$
J_\ell = D_\ell W_\ell,
$$

the column-scaled Wishart-L\'evy machinery is plausible after transposing. But
for the full network,

$$
J = D_L W_L \cdots D_1 W_1,
$$

the object of interest is

$$
J^\top J
=
W_1^\top D_1 \cdots W_L^\top D_L^2 W_L \cdots D_1 W_1,
$$

so one is dealing with a **product of alternating heavy-tailed random matrices
and random diagonal matrices**.

The main analytical issue is that the one-layer profile trick no longer closes:
after one multiplication, the singular vectors and the next diagonal factor are
correlated, so a single profile $c(u)$ is not enough.

## Approximation ladder

### 1. Mean-field independence closure

Replace each $D_\ell$ by an independent diagonal matrix whose entries are drawn
from the mean-field law of $|\phi'(h_\ell)|$.

Then each factor is an independent column-scaled heavy-tailed matrix.

### 2. Large-width product law

Try to describe the squared singular-value law of the product via a
multiplicative limit theorem:

- in the light-tailed case this is where free multiplicative convolution /
  $S$-transforms enter;
- in the heavy-tailed case, standard $S$-transform machinery is much less
  directly available, so one would likely need a **resolvent/Hermitization
  recursion** rather than moment methods.

### 3. Recursive cavity / fixed-point description

The most plausible heavy-tailed route is to derive a recursion for the
singular-value resolvent of the partial products

$$
J_{1:\ell}=D_\ell W_\ell J_{1:\ell-1}.
$$

That would be the analogue of Belinschi's one-layer fixed-point system, but now
for a sequence of effective singular-value laws.

### 4. Log-singular-value statistics

If one only wants bulk growth / contraction, a simpler target is the law of

$$
\log s_i(J),
$$

or just the top Lyapunov exponent. Then one can hope for an additive recursion
in log-scale, even when the full density is hard.

## Recommendation

- **For full densities:** aim for a **layer-by-layer resolvent recursion** under
  an independence ansatz for the $D_\ell$.
- **For more tractable analytics first:** target **tail exponents, survival
  asymptotics, and log-singular-value growth** rather than the full
  singular-value density.

In short, the full-network problem is not "column-scaled Wishart-L\'evy with a
more complicated profile"; it is a **heavy-tailed product-of-random-operators
problem**.
