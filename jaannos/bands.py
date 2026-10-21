"""Core + residue-band decomposition of one linear map, with a cheap sketched tail test.

    W = W_out + A_0 B_0 + sum_{j>=1} A_j B_j

A_j = U_j Sigma_j and B_j = V_j^T S^{-1} come from an SVD of (W - W_out) S, where S is the
Cholesky factor of the input second moment (identity if no calibration). Because the U_j are
orthonormal, the output error of stopping after band k is exactly

    || sum_{j>k} A_j B_j x ||^2 = sum_{j>k} || diag(sigma_j) B_j x ||^2,

and each band's term is estimated by a Gaussian sketch G_j = R_j diag(sigma_j) B_j (m rows).
"""

from __future__ import annotations

import numpy as np


class BandedLinear:
    def __init__(self, W, ranks, S=None, outlier_frac=0.0, m=32, seed=0):
        W = np.asarray(W, dtype=np.float64)
        self.d_out, self.d_in = W.shape
        rng = np.random.default_rng(seed)

        # outlier scalars ("super weights") kept in the always-on core
        W_out = np.zeros_like(W)
        n_out = int(round(outlier_frac * W.size))
        if n_out > 0:
            flat = np.argpartition(np.abs(W).ravel(), -n_out)[-n_out:]
            W_out.ravel()[flat] = W.ravel()[flat]
        self.W_out = W_out
        self.nnz_out = n_out

        Wr = W - W_out
        if S is None:
            M = Wr
        else:
            S = np.asarray(S, dtype=np.float64)
            M = Wr @ S
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        R = s.size
        ranks = list(ranks)
        if sum(ranks) != R:
            raise ValueError(f"ranks sum to {sum(ranks)}, full rank is {R}")
        self.ranks = ranks
        self.K = len(ranks) - 1                      # number of residue bands
        self.cum = np.cumsum(ranks)                  # rank prefix after band k (k = 0..K)

        self.A = U * s                               # (d_out, R)
        if S is None:
            self.B = Vt                              # (R, d_in)
        else:
            # B = Vt S^{-1}  <=>  S^T B^T = Vt^T
            self.B = np.linalg.solve(S.T, Vt.T).T
        self.s = s

        # one sketch per residue band; exact if the band is no wider than m
        self.G = []
        self.sketch_rows = []
        start = ranks[0]
        for r in ranks[1:]:
            SB = s[start:start + r, None] * self.B[start:start + r]
            if r <= m:
                G = SB
            else:
                G = rng.standard_normal((m, r)) / np.sqrt(m) @ SB
            self.G.append(G)
            self.sketch_rows.append(G.shape[0])
            start += r
        self.Gall = np.concatenate(self.G, axis=0)

        # multiply-add costs per token
        io = self.d_in + self.d_out
        self.dense_cost = self.d_in * self.d_out
        self.core_cost = ranks[0] * io + self.nnz_out
        self.band_cost = np.array([r * io for r in ranks[1:]], dtype=np.float64)
        self.sketch_cost = float(sum(self.sketch_rows) * self.d_in)
        self.prefix_cost = self.core_cost + np.concatenate([[0.0], np.cumsum(self.band_cost)])

    # ------------------------------------------------------------------ forward
    def forward(self, X, k):
        """X: (N, d_in). k: (N,) ints in 0..K, number of residue bands used per token."""
        X = np.asarray(X, dtype=np.float64)
        k = np.broadcast_to(np.asarray(k), (X.shape[0],))
        Z = X @ self.B.T                             # (N, R)
        cols = np.arange(self.B.shape[0])
        mask = cols[None, :] < self.cum[k][:, None]
        return (Z * mask) @ self.A.T + X @ self.W_out.T

    def core_forward(self, X):
        return self.forward(X, np.zeros(X.shape[0], dtype=int))

    def dense_equivalent(self):
        return self.A @ self.B + self.W_out

    # ------------------------------------------------------------------ tail tests
    def band_norms_exact(self, X):
        Z = np.asarray(X, dtype=np.float64) @ self.B.T
        out, start = [], self.ranks[0]
        for r in self.ranks[1:]:
            out.append(np.linalg.norm(Z[:, start:start + r] * self.s[start:start + r], axis=1))
            start += r
        return np.stack(out, axis=1)                 # (N, K)

    def band_norms_sketch(self, X):
        Y = np.asarray(X, dtype=np.float64) @ self.Gall.T
        out, start = [], 0
        for rows in self.sketch_rows:
            out.append(np.linalg.norm(Y[:, start:start + rows], axis=1))
            start += rows
        return np.stack(out, axis=1)

    def token_cost(self, k, sketch=False):
        k = np.asarray(k)
        return self.prefix_cost[k] + (self.sketch_cost if sketch else 0.0)


def tails_from_band_norms(norms):
    """norms (N, K) -> tails (N, K+1); tails[:, k] = norm of everything after band k."""
    sq = norms ** 2
    rev = np.cumsum(sq[:, ::-1], axis=1)[:, ::-1]    # rev[:, j] = sum_{i>=j} sq_i
    zeros = np.zeros((norms.shape[0], 1))
    return np.sqrt(np.concatenate([rev, zeros], axis=1))


def choose_k(tails, ref, eps):
    """Smallest k with tail_k <= eps * ref (per token)."""
    ok = tails <= eps * ref[:, None]
    return np.argmax(ok, axis=1)                     # tails[:, K] = 0, so always found
