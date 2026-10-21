# Preregistration: does a local, sketched residue test buy compute?

Written 2026-10-21, before any router result existed. The toy receipts and the GPT-2 receipts are
compared against this file, not against what looked good afterwards.

## The question

Can a frozen network decide, *locally* at each layer and for each token, how much of its own weights
it needs, using a test that is much cheaper than the computation it is deciding about, and beat
(a) spending the same compute uniformly and (b) one global confidence router?

The idea comes from the 2026-10-21 thread: dendritic "is the cheap answer enough?" tests, Sihti's
exact core + residue identity, NCM's coarse-to-fine hierarchy, and the 2024 "super weights" idea of
blocks summed on top of a base. The router trap identified in that thread is the cost inequality

    router wins only if   d < (1 - p) e

where d is the price of the test, e the price of the refinement it may skip, p the fraction of
cases that need it. A test that runs the refinement to find out whether the refinement is needed
can never win. The residue structure is used here only because it makes d small.

## Mechanism

Every linear map in the routed set is decomposed once, offline:

    W = W_out  +  A_0 B_0  +  sum_{j=1..K} A_j B_j

- `W_out`: sparse outlier scalars ("super weights", Yu et al. 2024) kept in the core, always on.
- `A_0 B_0`: the core band (rank r_0), always on.
- `A_j B_j`: residue bands (rank r_j each), from an SVD of `(W - W_out) S`, where `S` is the
  Cholesky factor of the input second moment (activation-aware; identity in the toy).
  `A_j = U_j Sigma_j`, `B_j = V_j^T S^{-1}`.

Because the `U_j` are orthonormal, the error of stopping after band k is exactly

    || sum_{j>k} A_j B_j x ||^2  =  sum_{j>k} || Sigma_j V_j^T S^{-1} x ||^2

and each term is estimated by a Johnson-Lindenstrauss sketch `G_j = R_j Sigma_j B_j` with m rows,
at price m * d_in per band instead of r_j * (d_in + d_out).

Sketch size, fixed before any router run: m = 4 rows per band. Reason, worked out while writing the
code (not after seeing results): the sketch price scales with d_in, and GPT-2's c_proj has
d_in = 3072. With m = 32 per band, the 8-band sketch of one c_proj costs 786k multiply-adds, a third
of the dense matrix, and the router is dead on arrival. With m = 4 it costs 98k (4%). A tail beyond
band k sums K - k band estimates, so it carries 4(K - k) degrees of freedom; the coarse decisions
(k near 0) are the well-measured ones.

## Routers (all evaluated at matched FLOPs, test cost included)

| name | decides | price it pays |
|---|---|---|
| `fixed` | one band count k for every token and layer (swept) | bands only |
| `local_sketch` | per layer, per token: smallest k with estimated relative tail <= eps (swept) | bands + sketch |
| `local_exact` | same rule with exact tail norms, test charged at zero | **oracle**, upper bound on the local signal, not a contender |
| `random_matched` | the exact k-arrays chosen by `local_sketch`, permuted across tokens within each layer | identical bill to `local_sketch` (bands + sketch), so only *where* the rank went differs |
| `global_margin` | pass 1 at k_min everywhere; tokens whose output top1-top2 margin < tau rerun with the original dense weights (tau and k_min swept, Pareto envelope taken) | pass 1 + dense price for flagged tokens |

Two choices above deliberately favour the attackers: `global_margin` reruns with dense weights
(cost 1.0) rather than the 1.25x full-rank factorization, and it gets the best envelope over
several k_min values. In GPT-2 it is also charged only once for unflagged tokens even though its
second pass recomputes them (their states shift because flagged neighbours change attention inputs).
| `dense` | the original weights | 1.0 by definition |

FLOPs are counted analytically (multiply-adds per token per routed matrix). Wall-clock is NOT
measured: per-token variable rank needs gather kernels that do not exist here, so the simulation
computes every band and masks. This is an accounting experiment.

Important consequence of factorization: a rank-r factor costs r (d_in + d_out). For GPT-2's
768 x 3072 MLP matrices, full rank costs 1.25x dense, and the break-even rank is 614 of 768.
Any router whose average rank exceeds break-even has already lost to `dense`.

## Metrics

Primary: top-1 agreement with the unmodified model (the model's own answer, not the label),
read off each router's curve at matched relative cost budgets.
Secondary: mean KL(dense || routed); NLL on true labels/next tokens.

Matched comparison: each router's swept points are reduced to their Pareto frontier
(cost down, agreement up), and the frontier is interpolated piecewise-linearly at the budgets
below (interpolating between two settings = using each on a fraction of tokens, which is
realizable). Above a router's largest cost, its best point is used. Below its smallest cost it
reports NaN.

## Gates

### G0 identity (toy, must pass or nothing else is read)
- Full reconstruction: `W_out + sum_all A_j B_j == W` to relative Frobenius error < 1e-10 (float64).
- With all bands on, routed output == dense output to 1e-9.
- Sketch: on a random 768 x 3072 matrix with the GPT-2 band layout (core 256, 8 x 64) and m = 4
  per band, the full-tail estimate (k = 0, 32 rows in total) has median | est / true - 1 | < 0.15
  over random inputs. (Relative std of a 32-row Gaussian sketch of a norm is about 0.125.)

### G1 accounting (toy)
- Router-reported FLOPs equal an independent recount from the k-arrays.
- `random_matched` cost equals `local_sketch` cost exactly for every eps.

### G2 null world (toy) — the free control
Isotropic Gaussian inputs, no designed difficulty structure, spectra decaying smoothly.
Prediction: `local_sketch` does NOT beat `fixed` by more than 1.0 point of agreement at any budget
in {0.5, 0.6, 0.7, 0.8}. If it does, the toy or the accounting is leaking and nothing downstream is
trusted.

### G3 constructed world (toy) — mechanism check, NOT evidence
A fraction p = 0.2 of inputs carries large energy along the tail input directions of the first
layer. This world is built so that a local residue test *can* see difficulty. Reported only to show
the machinery does what it says. A local win here is expected and proves nothing about real models.
A local loss here would mean the implementation is broken.

### G4 GPT-2 (run on the user's GPU) — the actual test
Routed set: all 24 MLP matrices (c_fc, c_proj in 12 blocks). Attention untouched.
Bands: core rank 256, then 8 bands of 64 (total 768). Sketch m = 4 per band. Outliers: top 1e-4 of |w|.
Calibration: 32k tokens of WikiText-2 train. Evaluation: WikiText-2 test, 16 x 512 tokens.
Budgets (fraction of dense MLP FLOPs): {0.6, 0.7, 0.8, 0.9}.

PASS (local signal is useful) requires ALL of:
1. `local_sketch` beats the better of `fixed` and `global_margin` by >= 1.0 point top-1 agreement
   at >= 2 of the 4 budgets, and is not worse than either by > 1.0 point at any budget.
2. `random_matched` loses to `local_sketch` by >= 1.0 point at those same budgets
   (the win is the allocation, not the budget distribution).
3. The `local_sketch` point used at each budget lies below cost 1.0 (beats dense on price).

KILL: `fixed` ties or beats `local_sketch` (within 1.0 point) at all four budgets. That closes the
"local residue test buys compute" claim for this model family, and the result is kept.

Recorded but not a gate: gap between `local_sketch` and `local_exact` (what the sketch costs in
signal quality), and whether any router reaches 99% agreement below dense cost at all.

## My prior, stated before running

GPT-2 MLP weight spectra are fairly flat even after whitening (SVD-compression papers lose quality
fast). I expect the high-agreement region to sit near or above the 614 break-even rank, where nobody
beats `dense`, and I expect `fixed` to be hard to beat in the 0.6-0.9 band. My guess is KILL or a
narrow win at the lowest budget only. The experiment is worth running because the cost argument is
now right; the prior is not a reason to skip it.
