import numpy as np
import pytest

from jaannos.bands import BandedLinear, choose_k, tails_from_band_norms
from jaannos.routing import Fixed, Local, Replay, ToyNet, at_budget, pareto, permute_k


def test_reconstruction_and_full_rank_identity():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((40, 24))
    X = rng.standard_normal((50, 24))
    L = np.linalg.cholesky(X.T @ X / 50 + 1e-3 * np.eye(24))
    b = BandedLinear(W, [8, 8, 8], S=L, outlier_frac=0.01, m=3)
    assert np.linalg.norm(b.dense_equivalent() - W) / np.linalg.norm(W) < 1e-12
    assert np.allclose(b.forward(X, np.full(50, b.K)), X @ W.T, atol=1e-10)


def test_exact_tail_equals_truncation_error():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((30, 20))
    X = rng.standard_normal((10, 20))
    L = np.linalg.cholesky(np.cov(rng.standard_normal((200, 20)).T) + 1e-3 * np.eye(20))
    b = BandedLinear(W, [5, 5, 5, 5], S=L)
    t = tails_from_band_norms(b.band_norms_exact(X))
    full = b.forward(X, np.full(10, b.K))
    for k in range(b.K + 1):
        err = np.linalg.norm(full - b.forward(X, np.full(10, k)), axis=1)
        assert np.allclose(err, t[:, k], rtol=1e-9, atol=1e-10)


def test_choose_k_is_smallest_satisfying():
    tails = np.array([[3.0, 2.0, 1.0, 0.0], [0.5, 0.2, 0.1, 0.0]])
    k = choose_k(tails, np.array([10.0, 10.0]), 0.15)
    assert list(k) == [2, 0]


def test_cost_recount_and_matched_bill():
    net = ToyNet(d=32, d_ff=64, L=2, ranks=[8] + [4] * 6, m=2, seed=3)
    X = np.random.default_rng(4).standard_normal((100, 32))
    _, c, ks = net.run(X, Local(0.05))
    recount = sum(net.blin[i].token_cost(k, sketch=True) for i, k in enumerate(ks))
    assert np.array_equal(recount, c)
    _, cr, _ = net.run(X, Replay(permute_k(ks, 0)))
    assert np.isclose(cr.sum(), c.sum())
    _, cf, _ = net.run(X, Fixed(0))
    assert np.allclose(cf, sum(b.core_cost for b in net.blin))


def test_pareto_and_budget_interpolation():
    pts = [(0.5, 0.90), (0.6, 0.89), (0.7, 0.95), (0.9, 0.99)]
    assert pareto(pts) == [(0.5, 0.90), (0.7, 0.95), (0.9, 0.99)]
    assert np.isclose(at_budget(pts, 0.6), 0.925)
    assert np.isnan(at_budget(pts, 0.4))
    assert at_budget(pts, 1.5) == 0.99


def test_torch_banded_matches_dense():
    torch = pytest.importorskip("torch")
    import gpt2_gate as g
    rng = np.random.default_rng(5)
    W = torch.tensor(rng.standard_normal((48, 16)))
    Xc = torch.tensor(rng.standard_normal((200, 16)))
    C = Xc.T @ Xc / 200
    b = g.Banded(W, C, core=4, band=4, m=2, outlier_frac=0.01, ridge=1e-4, seed=0, device="cpu")
    X = torch.tensor(rng.standard_normal((7, 16)), dtype=torch.float32)
    full = b.forward(X, torch.full((7,), b.K))
    assert torch.allclose(full, X @ W.float().T, atol=1e-4)
    t = g.tails(b.band_norms(X, exact=True))
    err = torch.linalg.norm(full - b.forward(X, torch.zeros(7, dtype=torch.long)), dim=1)
    assert torch.allclose(err.double(), t[:, 0], rtol=1e-3, atol=1e-4)
