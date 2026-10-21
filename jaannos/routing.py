"""Routers, a small residual MLP to route through, and matched-budget comparison."""

from __future__ import annotations

import numpy as np

from .bands import BandedLinear, choose_k, tails_from_band_norms


# ---------------------------------------------------------------- policies
class Fixed:
    name = "fixed"

    def __init__(self, k):
        self.k = k

    def __call__(self, idx, blin, X):
        k = np.full(X.shape[0], min(self.k, blin.K), dtype=int)
        return k, blin.token_cost(k)


class Local:
    """Per-token, per-matrix: smallest k whose estimated tail <= eps * ||core output||."""

    def __init__(self, eps, exact=False):
        self.eps = eps
        self.exact = exact
        self.name = "local_exact" if exact else "local_sketch"

    def __call__(self, idx, blin, X):
        norms = blin.band_norms_exact(X) if self.exact else blin.band_norms_sketch(X)
        ref = np.linalg.norm(blin.core_forward(X), axis=1) + 1e-12
        k = choose_k(tails_from_band_norms(norms), ref, self.eps)
        # the oracle is charged nothing for its test
        return k, blin.token_cost(k, sketch=not self.exact)


class Replay:
    """Use recorded k-arrays (e.g. permuted ones) and pay the same bill as the recording."""

    name = "random_matched"

    def __init__(self, k_list, sketch=True):
        self.k_list = k_list
        self.sketch = sketch

    def __call__(self, idx, blin, X):
        k = self.k_list[idx]
        return k, blin.token_cost(k, sketch=self.sketch)


def permute_k(k_list, seed):
    rng = np.random.default_rng(seed)
    return [rng.permutation(k) for k in k_list]


# ---------------------------------------------------------------- toy model
def random_orthogonal(n, rng):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


def spectral_matrix(d_out, d_in, alpha, top, rng, n_outliers=0, outlier_scale=8.0):
    """Matrix with singular values top * (i+1)^-alpha plus a few large outlier scalars."""
    R = min(d_out, d_in)
    U = random_orthogonal(d_out, rng)[:, :R]
    V = random_orthogonal(d_in, rng)[:, :R]
    s = top * (np.arange(R) + 1.0) ** (-alpha)
    W = (U * s) @ V.T
    if n_outliers:
        idx = rng.choice(W.size, n_outliers, replace=False)
        W.ravel()[idx] += rng.choice([-1, 1], n_outliers) * outlier_scale * W.std()
    return W


class ToyNet:
    """h <- h + W2 relu(W1 h), L blocks, linear readout. Every W1, W2 is routed."""

    def __init__(self, d=128, d_ff=256, L=4, C=10, alpha=1.0, ranks=None, m=3,
                 outlier_frac=2e-4, seed=0):
        rng = np.random.default_rng(seed)
        self.d, self.d_ff, self.L, self.C = d, d_ff, L, C
        R = min(d, d_ff)
        self.ranks = ranks or [32] + [12] * 8
        assert sum(self.ranks) == R
        self.W = []
        self.blin = []
        for l in range(L):
            W1 = spectral_matrix(d_ff, d, alpha, 2.0, rng, n_outliers=3)
            W2 = spectral_matrix(d, d_ff, alpha, 0.6, rng, n_outliers=3)
            self.W += [W1, W2]
            self.blin += [BandedLinear(W1, self.ranks, outlier_frac=outlier_frac, m=m, seed=seed + 2 * l),
                          BandedLinear(W2, self.ranks, outlier_frac=outlier_frac, m=m, seed=seed + 2 * l + 1)]
        self.Wc = rng.standard_normal((C, d)) / np.sqrt(d)
        self.dense_cost_per_token = float(sum(b.dense_cost for b in self.blin))

    def dense(self, X):
        h = X
        for l in range(self.L):
            h = h + np.maximum(h @ self.W[2 * l].T, 0) @ self.W[2 * l + 1].T
        return h @ self.Wc.T

    def run(self, X, policy):
        """Returns logits, per-token cost (multiply-adds), list of k-arrays per routed matrix."""
        h = X
        cost = np.zeros(X.shape[0])
        ks = []
        for l in range(self.L):
            for j in (0, 1):
                idx = 2 * l + j
                b = self.blin[idx]
                inp = h if j == 0 else a
                k, c = policy(idx, b, inp)
                ks.append(k)
                cost += c
                out = b.forward(inp, k)
                if j == 0:
                    a = np.maximum(out, 0)
                else:
                    h = h + out
        return h @ self.Wc.T, cost, ks


# ---------------------------------------------------------------- metrics
def margin(logits):
    top2 = np.sort(logits, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def score(logits, dense_logits):
    agree = float(np.mean(logits.argmax(1) == dense_logits.argmax(1)))
    p, q = softmax(dense_logits), softmax(logits)
    kl = float(np.mean(np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)), axis=1)))
    return agree, kl


def global_margin(net, X, dense_logits, k_min, tau):
    logits1, cost1, _ = net.run(X, Fixed(k_min))
    flag = margin(logits1) < tau
    logits = np.where(flag[:, None], dense_logits, logits1)
    cost = cost1 + flag * net.dense_cost_per_token
    return logits, cost, float(flag.mean())


def pareto(points):
    """points: list of (cost, agree). Frontier with cost ascending, agreement strictly rising."""
    pts = sorted(points, key=lambda t: (t[0], -t[1]))
    out, best = [], -np.inf
    for c, a in pts:
        if a > best:
            out.append((c, a))
            best = a
    return out


def at_budget(points, b):
    f = pareto(points)
    if not f or b < f[0][0]:
        return float("nan")
    if b >= f[-1][0]:
        return f[-1][1]
    for (c0, a0), (c1, a1) in zip(f, f[1:]):
        if c0 <= b <= c1:
            return a0 + (a1 - a0) * (b - c0) / (c1 - c0)
    return float("nan")
