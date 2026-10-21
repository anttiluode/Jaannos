# Jäännös

*Jäännös* is Finnish for "residue": what is left after the core has been taken.

This repo asks one question that came out of the 2026-10-21 thread (dCaAP XOR neurons, Sihti's
exact core + residue identity, NCM's coarse-to-fine hierarchy, and the 2024 "super weights" idea of
blocks summed on top of a base):

> Can a frozen network decide **locally**, per layer and per token, how much of its own weights it
> needs, using a test much cheaper than the computation it is deciding about, and beat both
> spending the same compute uniformly and one global confidence router?

The preregistration in `docs/PREREG.md` was written before any router result existed.

## The mechanism

Each routed matrix is decomposed once, offline:

```
W = W_out + A_0 B_0 + Σ_{j=1..K} A_j B_j
```

- `W_out` holds sparse outlier scalars, always on.
- `A_0 B_0` is the core rank band, always on.
- `A_j B_j` are the residue bands, taken from an SVD of `(W − W_out) S`, where `S` is the Cholesky
  factor of the input second moment (activation-aware).

Because the left singular vectors are orthonormal, the output error of stopping after band k is exact:

```
‖Σ_{j>k} A_j B_j x‖² = Σ_{j>k} ‖diag(σ_j) B_j x‖²
```

A Gaussian sketch with m = 4 rows per band estimates each term at a cost of m·d_in. That is the
cheap test the router trap required, `d < (1 − p) e`. The local router picks, for each token, the
smallest k whose estimated tail is at most ε times the norm of the core output.

The attackers, all at matched multiply-add cost with the test included:

- **fixed**: the same k for every token, swept.
- **random_matched**: the local router's own k-allocations shuffled across tokens. Same bill; only
  where the rank went differs.
- **global_margin**: a cheap pass, then dense weights for tokens whose top1−top2 output margin is
  small. It is deliberately generous: true dense logits, dense price paid once.
- **local_exact**: the same local rule with exact tails, charged nothing for the test. This is an
  oracle that bounds what any local residue test could achieve.

## Results so far (toy, CPU, `python run_toy.py`)

The toy is a 4-block residual MLP with d = 128 and d_ff = 256. All 8 matrices are routed.
The core band costs 0.375 of dense, each band 0.14, and the sketch 0.14.

**G0 identity: PASS.**
- Reconstruction error is 9e-15 and all-bands output matches dense to 2e-15.
- On GPT-2-shaped random matrices (core 256 + 8×64, m = 4), the sketch's full-tail median relative
  error is 0.087 for 3072×768 and 0.088 for 768×3072.
- On those shapes the sketch costs 1.0% and 4.2% of dense.

**G1 accounting: PASS.** An independent recount matches exactly, and random_matched pays exactly
the local router's bill.

**G2 null world (isotropic inputs), top-1 agreement with dense, %:**

| router | b=0.5 | b=0.6 | b=0.7 | b=0.8 |
|---|---|---|---|---|
| fixed | 92.99 | 93.30 | 93.70 | 94.21 |
| local_sketch | — | 93.13 | 93.43 | 93.73 |
| local_exact (oracle) | 93.06 | 93.33 | 93.70 | 94.14 |
| random_matched | — | 92.80 | 93.13 | 93.45 |
| **global_margin** | **97.29** | **99.14** | **99.84** | **100.00** |

G2 passes: local does not beat fixed on noise, so the accounting has no leak.

**G3 constructed world (mechanism check, not evidence):** the same pattern. global_margin scores
97.45 / 99.14 / 99.64 / 99.85, and local_sketch is within about 0.6 points of fixed in either
direction. The diagnostics are the interesting part:

- The local router did what it was built for. At layer 0 it gave the designed-hard tokens k = 8.0
  bands against 3.5 for the easy ones, and the sketch's tail ranking correlates with the exact
  tail ranking at Spearman 0.69.
- But those tokens were not the ones whose answers flipped. Core-only disagreement was 6.5% on hard
  tokens and 7.4% on easy ones.
- So the construction of a hard/easy world failed. A large residue did not make an answer wrong.

**What the toy says.** Size of the residue is not decision relevance.
- Truncation error spreads thinly over all tokens. Fixed rank buys only about 0.4 points per 0.14
  of cost.
- The argmax only flips on tokens that are already near their decision boundary.
- The global margin sees the boundary. A local residue norm cannot, even the exact oracle.
- Rerunning the 5% lowest-margin tokens densely lifts agreement from 92.5% to 95.0% for 0.05 of
  cost. Adding all eight bands to every token costs 1.125 more and reaches 100%.

This is the same correction Sol made to the dCaAP reading (threshold proximity is not class margin),
now measured. It is also V24 Gate 6B again: signal magnitude was not "expected uncertainty removed".

The first toy configuration (spectrum decay α = 1.0) was uninformative, because core-only already
agreed 99.7%. Its receipt is kept as `results/toy_receipt_v0_ceiling.json`, and v1 uses α = 0.35.
The constructed-world amplitude was kept at the original value rather than tuned.

## The real test: GPT-2 (G4, run on your GPU)

```
pip install -r requirements.txt
python gpt2_gate.py                 # GPT-2 small, WikiText-2; routes all 24 MLP matrices
python gpt2_gate.py --smoke         # tiny random model, no downloads, code-path check only
```

- It writes `results/gpt2_receipt.json` and prints the matched-budget table and the preregistered
  verdict (PASS / KILL / NEITHER).
- Settings: core rank 256 plus 8 bands of 64, m = 4, calibration on 32k WikiText-2 train tokens,
  evaluation on 16×512 test tokens, budgets of 0.6 / 0.7 / 0.8 / 0.9 of dense MLP cost.
- Break-even matters here. A rank-r factor of a 768×3072 matrix costs r·3840. Full rank is 1.25×
  dense, and anything above rank 614 has already lost to just using the original weights.
- `--text-file your.txt` replaces WikiText-2 (80/20 calibration/evaluation split).
- It has only been smoke-tested here, because model downloads are blocked in this sandbox. Expect a
  few minutes on a GPU, mostly the 24 decompositions and roughly 60 evaluation sweeps.

The prior, stated in the prereg and now sharpened by the toy: **KILL is the expected outcome.**
The toy suggests global_margin will dominate, because LLM top-1 flips also concentrate at
low-margin positions. A PASS on GPT-2 would therefore be a genuine surprise, and worth a lot.

## Honest boundaries

- **Accounting, not speed.** All bands are computed and then masked. Per-token variable rank needs
  gather kernels that are not written here.
- **Prior art on the weight side is occupied.** Nested and any-precision weights already exist
  (Any-Precision LLM, Matryoshka Quantization, MatFormer), and activation-aware SVD is
  ASVD / SVD-LLM. The untested piece is only the per-token sketched-residue stopping rule.
- **Not tested, and not preregistered:** local + margin hybrids, and learned routers
  (mixture-of-depths). If G4 is KILL, the toy already points at why. The decision signal lives at
  the output boundary, not in local residue magnitude.

## Layout

```
docs/PREREG.md          gates, attackers, pass/kill rules, prior — written before results
jaannos/bands.py        BandedLinear: core + bands + outliers, exact tails, sketch, costs (numpy)
jaannos/routing.py      routers, toy residual MLP, matched-budget Pareto comparison
run_toy.py              G0-G3 -> results/toy_receipt.json
gpt2_gate.py            G4 on GPT-2 (torch) -> results/gpt2_receipt.json
tests/                  6 tests: identities, exact-tail = truncation error, accounting, torch parity
results/                toy_receipt.json (v1), toy_receipt_v0_ceiling.json, gpt2_smoke_receipt.json
```
