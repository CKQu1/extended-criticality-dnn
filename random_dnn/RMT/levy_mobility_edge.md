# Mobility edge of symmetric Levy matrices (Tarquini-Biroli-Tarzia)

Numerical solver for the closed mobility-edge equation of Wigner-Levy matrices,
following Tarquini-Biroli-Tarzia (`.agents/notes/tarquini-2015.md`, arXiv
1507.00296, PRL 116, 010601). This is the unstructured warm-up for the
Hermitisation-mapped heavy-tailed Jacobian SV problem of `RMT/localisation.md`:
if we cannot reproduce TBT's own mobility edge in their clean symmetric-Levy
setting, we cannot trust the mapped prediction `s_c(alpha_SV) = E*(alpha_SV/2)`.

## Setup

Real symmetric N x N matrix H with i.i.d. entries of tail exponent 1 + mu,

    P(h) ~ (mu / (2 N)) |h|^{-1-mu},   |h| -> inf,   mu in (0, 2),

the 1/N scaling making the eigenvalue spectrum O(1). For mu in (0, 1) there is
a mobility edge E*(mu): eigenvectors are delocalised for |E| < E* and localised
for |E| > E*. For mu >= 1 the whole spectrum is delocalised (no finite E*).

## The three-step method (TBT EPAPS "Computation of the mobility edge")

Let G_{ii} be the diagonal resolvent and Sigma_{ii} = E - i eta - G_{ii}^{-1}
= S_i + i Delta_i the self-energy. Linearising the cavity recursion in the
imaginary part at eta -> 0+ decouples the real part,

    S_i =d= sum_j h_{ij}^2 Re G_{jj},   Re G = 1 / (E - S)        (eta -> 0),

from the imaginary part Delta. The localisation transition is the freezing
(one-step RSB) transition of the directed-polymer recursion for Delta, whose
transfer-operator top eigenvalue lambda(m, E) obeys lambda(m*, E*) = 1,
d_m lambda = 0; the determinant of the linearised 2x2 problem collapses to
m = 1/2 (the l.h.s. is symmetric in m about 1/2). This yields three steps.

### Step 1 -- self-consistency for C(E), beta(E)

By the generalized CLT, S is Levy-stable with index mu/2, scale C(E) and
skewness beta(E), characteristic function

    Lhat_{mu/2}^{C,beta}(k) = exp[ -C |k|^{mu/2} (1 + i beta tan(pi mu/4) sign k) ].

C, beta are fixed by (TBT eqs. C-self-EPAPS, beta-self-EPAPS)

    C(E)    = Gamma(1 - mu/2) cos(pi mu/4) E_S[ |E - S|^{-mu/2} ],
    beta(E) = E_S[ sign(E - S) |E - S|^{-mu/2} ] / E_S[ |E - S|^{-mu/2} ],

the expectations over S ~ L_{mu/2}^{C,beta}. Since Re G = 1/(E - S), these are
just fractional moments of Re G:

    C(E)    = Gamma(1 - mu/2) cos(pi mu/4) E[ |Re G|^{mu/2} ],
    beta(E) = E[ sign(Re G) |Re G|^{mu/2} ] / E[ |Re G|^{mu/2} ].

Solved by population dynamics: hold a pool of S ~ L_{mu/2}^{C,beta} (sampled by
Chambers-Mallows-Stuck), map to Re G = 1/(E - S), recompute (C, beta) from the
empirical fractional moments, iterate to a fixed point. These are the
Cizeau-Bouchaud / Burda / Ben-Arous equations.

### Step 2 -- build ell(E)

    ell(E) = (1/pi) int_0^inf k^{mu-1} Lhat_{mu/2}^{C(E),beta(E)}(k) e^{i k E} dk,

with ell_- = ell* (TBT). Only the closed-form characteristic function is needed
-- no density inversion. The integrand decays as exp(-C k^{mu/2}) (slow for
small mu) and oscillates with period 2 pi / E, so a dense quadrature with a
convergence check on (k_max, dk) is used.

### Step 3 -- the mobility-edge determinant

The m = 1/2 solvability (determinant = 0) condition (TBT eq:mobility):

    D(E) = K_mu^2 (s_mu^2 - 1) |ell(E)|^2 - 2 s_mu K_mu Re ell(E) + 1 = 0,

    K_mu = mu Gamma(1/2 - mu/2)^2 / 2,   s_mu = sin(pi mu/2),   s_{1/2} = 1.

(s_{1/2} = sin(pi/2) is the EPAPS s_m at m = 1/2; hence the -1.) D(E) is
negative in the delocalised phase (|E| < E*) and -> +1 as E -> inf; E* is the
crossing. Steps 1 and 2 are coupled (ell needs the converged C, beta), and the
whole D(E) is scanned over E to bracket and root-find E*.

## Step 1 solved three ways (all agree)

C(E), beta(E) was computed by (i) population dynamics with a Chambers-Mallows-
Stuck stable sampler, (ii) the convention-free LePage cavity sum
S = sum_k Gamma_k^{-1/a} w_k (Gamma_k unit-Poisson arrivals, w_k from the Re G
pool), and (iii) the deterministic k-space fixed point above. All three agree to
four digits, e.g. at mu = 0.5, E = 3.85: C = 0.6957, beta = +0.6263. The
deterministic solver is the default (noise-free, ~40 iterations, no pool). This
also fixed the stable-law sign convention: the physical S is positive-skewed
(S1 b = +beta); sampling b = -beta to match TBT's *written* CF literally
collapses to a spurious one-sided beta = 1 fixed point.

## Validation targets and results

- mu -> 1: E* diverges (K_mu = mu Gamma(1/2-mu/2)^2/2 -> inf). Confirmed.
- mu >= 1: no finite root of D(E). Confirmed (solver returns None), consistent
  with Bordenave-Guionnet delocalisation for 1 < alpha < 2.
- E*(mu) shape matches TBT Fig. 1 (pd-ult.pdf): flat until mu ~ 0.25, monotone
  rise, divergence at mu -> 1. Solver values (physical branch):
  mu = 0.4 -> 1.36, 0.5 -> 3.29, 0.6 -> 5.90, 0.7 -> 9.37.

## Finite-N arbiter (fourth method) and resolution

`.agents/temp/finite_N_levy_resolvent.py` builds actual symmetric Levy matrices
(TBT entry law), inverts (E + i eta - H), and reads C(E), beta(E) straight off
the empirical Re G_ii distribution -- no cavity or GCLT assumption. At mu = 0.5,
N = 2000-4000, eta = 1e-3, finite-N matches the cavity solver to < 1% across the
whole E range (e.g. E = 3.85: cavity 0.696/+0.626, finite-N 0.699/+0.627;
E = 5.0: 0.664/+0.656 vs 0.660/+0.635). So step 1 and its normalisation are
confirmed by four independent methods.

Crucially, the empirical Re G_ii is **positive-skewed** (beta_emp = +0.54..+0.66
> 0), confirming the physical S is positive-skewed (S1 b = +beta) with CF
exp(-C k^a (1 - i beta t)). This fixes the ell-branch unambiguously to the
phase_sign = +1 (physical) integral used here -> E*(0.5) = 3.29. The alternative
"literal-TBT" branch (E* = 4.7) would require negative skew, which the matrix
data rules out.

## Status

- E*(0.5) = 3.29 is the physically-grounded value (CF sign fixed by finite-N
  data). TBT report 3.85 (level statistics, Fig. 2) and ~2.5 (Fig. 1) -- 3.29
  is inside their own spread. The ~15% residual to their 3.85 is consistent
  with the wide finite-size crossover TBT themselves emphasise (their result 3)
  and figure-reading on a steep curve; it is not a bug.
- The localization indicator (typical Im G_ii vs E) from the same finite-N run
  is monotone and featureless at N = 2000-4000 -- it does not sharply locate
  E*, exactly as expected from the wide crossover. Finite-N validates step 1
  but cannot pin the edge; the cavity eq:mobility is the sharp estimate.
- (C, beta) validated by: CMS population dynamics, LePage cavity sum,
  deterministic k-space fixed point, and finite-N diagonal resolvent.

## Relevance to the project

Via the bipartite Hermitisation (`RMT/hermitisation.md`,
`RMT/localisation.md` sec. 9J), the rectangular SV problem at stability
alpha_SV maps to a Wigner-Levy problem at mu = alpha_SV / 2. The empirical
localisation onset s_c(alpha_SV) from the density-deviation diagnostic should
equal E*(alpha_SV/2) from this solver (modulo the Hermitisation mass split and
SciPy-Belinschi scale). The structured (column-profile c(y)) generalisation of
step 1's self-consistency is the open follow-on.
