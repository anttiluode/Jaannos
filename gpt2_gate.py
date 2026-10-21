"""G4 from docs/PREREG.md: local sketched residue routing on GPT-2's MLPs, at matched FLOPs.

    pip install torch transformers datasets
    python gpt2_gate.py                       # GPT-2 small, WikiText-2, CUDA if available
    python gpt2_gate.py --text-file my.txt    # any text instead of WikiText-2 (80/20 calib/eval split)
    python gpt2_gate.py --smoke               # tiny random model, no downloads, checks the code paths

Writes results/gpt2_receipt.json and prints the matched-budget table plus the preregistered verdict.

What is routed: every mlp.c_fc and mlp.c_proj (24 matrices in GPT-2 small). Attention is untouched.
Costs are analytic multiply-adds over the routed matrices, as a fraction of their dense cost.
Wall-clock is not measured: all bands are computed and masked (see PREREG).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

BUDGETS = [0.6, 0.7, 0.8, 0.9]


# ======================================================================= banded matrix (torch)
class Banded:
    """W (out, in) = W_out + A_0 B_0 + sum_j A_j B_j, with activation-aware SVD and a tail sketch."""

    def __init__(self, W, C, core, band, m, outlier_frac, ridge, seed, device):
        W = W.to(device, torch.float64).contiguous()
        d_out, d_in = W.shape
        R = min(d_out, d_in)
        ranks = [min(core, R)]
        while sum(ranks) < R:
            ranks.append(min(band, R - sum(ranks)))
        self.ranks, self.K = ranks, len(ranks) - 1
        self.cum = torch.tensor(np.cumsum(ranks), device=device)

        # outliers ("super weights"), scored by |w_ij| * rms(x_j)
        n_out = int(round(outlier_frac * W.numel()))
        W_out = torch.zeros_like(W)
        if n_out > 0:
            score = W.abs() * torch.sqrt(torch.clamp(torch.diagonal(C), min=0))[None, :]
            idx = torch.topk(score.flatten(), n_out).indices
            W_out.view(-1)[idx] = W.view(-1)[idx]
        self.nnz_out = n_out

        Cr = C + ridge * torch.diagonal(C).mean() * torch.eye(d_in, device=device, dtype=torch.float64)
        S = torch.linalg.cholesky(Cr)                         # C = S S^T
        U, s, Vt = torch.linalg.svd((W - W_out) @ S, full_matrices=False)
        B = torch.linalg.solve_triangular(S.T, Vt.T, upper=True).T     # B = Vt S^{-1}
        A = U * s

        gen = torch.Generator(device="cpu").manual_seed(seed)
        G, rows = [], []
        start = ranks[0]
        for r in ranks[1:]:
            SB = s[start:start + r, None] * B[start:start + r]
            if r <= m:
                Gj = SB
            else:
                Rm = torch.randn(m, r, generator=gen, dtype=torch.float64).to(device) / math.sqrt(m)
                Gj = Rm @ SB
            G.append(Gj)
            rows.append(Gj.shape[0])
            start += r
        self.sketch_rows = rows

        f32 = lambda t: t.to(torch.float32).contiguous()
        self.A, self.B, self.W_out, self.s = f32(A), f32(B), f32(W_out), f32(s)
        self.G = f32(torch.cat(G, 0))
        self.recon_rel_err = float(torch.linalg.norm(A @ B + W_out - W) / torch.linalg.norm(W))

        io = d_in + d_out
        self.dense_cost = float(d_in * d_out)
        self.core_cost = float(ranks[0] * io + n_out)
        band_cost = [float(r * io) for r in ranks[1:]]
        self.sketch_cost = float(sum(rows) * d_in)
        self.prefix_cost = torch.tensor([self.core_cost + sum(band_cost[:k]) for k in range(self.K + 1)],
                                        device=device, dtype=torch.float64)

    def forward(self, X, k):
        Z = X @ self.B.T
        cols = torch.arange(Z.shape[1], device=X.device)
        mask = (cols[None, :] < self.cum[k][:, None]).to(Z.dtype)
        return (Z * mask) @ self.A.T + X @ self.W_out.T

    def band_norms(self, X, exact):
        out = []
        if exact:
            Z = X @ self.B.T
            start = self.ranks[0]
            for r in self.ranks[1:]:
                out.append(torch.linalg.norm(Z[:, start:start + r] * self.s[start:start + r], dim=1))
                start += r
        else:
            Y = X @ self.G.T
            start = 0
            for rows in self.sketch_rows:
                out.append(torch.linalg.norm(Y[:, start:start + rows], dim=1))
                start += rows
        return torch.stack(out, 1)

    def token_cost(self, k, sketch):
        return self.prefix_cost[k] + (self.sketch_cost if sketch else 0.0)


def tails(norms):
    sq = norms.double() ** 2
    rev = torch.flip(torch.cumsum(torch.flip(sq, [1]), 1), [1])
    return torch.sqrt(torch.cat([rev, torch.zeros_like(rev[:, :1])], 1))


# ======================================================================= routing controller
class Controller:
    def __init__(self):
        self.policy = None
        self.i = 0
        self.ks = []
        self.cost = None

    def start(self, policy, n_tokens, device):
        self.policy, self.i, self.ks = policy, 0, []
        self.cost = torch.zeros(n_tokens, dtype=torch.float64, device=device)


class RoutedConv1D(nn.Module):
    """Drop-in for transformers' Conv1D (y = x @ W + b, W is (in, out))."""

    def __init__(self, conv, banded, ctrl):
        super().__init__()
        self.weight, self.bias = conv.weight, conv.bias
        self.banded, self.ctrl = banded, ctrl

    def forward(self, x):
        ctrl = self.ctrl
        if ctrl.policy is None:
            return x @ self.weight + self.bias
        shp = x.shape
        X = x.reshape(-1, shp[-1])
        k, cost = ctrl.policy(ctrl.i, self.banded, X)
        ctrl.ks.append(k)
        ctrl.cost += cost
        ctrl.i += 1
        y = self.banded.forward(X, k) + self.bias
        return y.reshape(*shp[:-1], -1)


class Fixed:
    def __init__(self, k):
        self.k = k

    def __call__(self, i, b, X):
        k = torch.full((X.shape[0],), min(self.k, b.K), dtype=torch.long, device=X.device)
        return k, b.token_cost(k, False)


class Local:
    def __init__(self, eps, exact):
        self.eps, self.exact = eps, exact

    def __call__(self, i, b, X):
        t = tails(b.band_norms(X, self.exact))
        ref = torch.linalg.norm(b.forward(X, torch.zeros(X.shape[0], dtype=torch.long, device=X.device)), dim=1).double()
        k = torch.argmax((t <= self.eps * (ref[:, None] + 1e-12)).int(), dim=1)
        return k, b.token_cost(k, not self.exact)


class Replay:
    def __init__(self, ks):
        self.ksq = ks

    def __call__(self, i, b, X):
        k = self.ksq[i]
        return k, b.token_cost(k, True)


# ======================================================================= data
def load_ids(tok, args):
    if args.text_file:
        text = open(args.text_file, encoding="utf-8").read()
        cut = int(0.8 * len(text))
        tr, te = text[:cut], text[cut:]
    else:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        tr, te = "\n\n".join(ds["train"]["text"]), "\n\n".join(ds["test"]["text"])
    tr_ids = tok(tr[:2_000_000], return_tensors="pt").input_ids[0]
    te_ids = tok(te, return_tensors="pt").input_ids[0]
    L = args.seq_len
    calib = tr_ids[: (args.calib_tokens // L) * L].reshape(-1, L)
    ev = te_ids[: args.eval_seqs * L].reshape(-1, L)
    return calib, ev


# ======================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--calib-tokens", type=int, default=32768)
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--core", type=int, default=256)
    ap.add_argument("--band", type=int, default=64)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--outlier-frac", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--n-eps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny random GPT-2, random tokens, no downloads")
    ap.add_argument("--out", default="results/gpt2_receipt.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    t0 = time.time()

    from transformers import GPT2Config, GPT2LMHeadModel
    if args.smoke:
        cfg = GPT2Config(n_embd=64, n_layer=2, n_head=2, n_positions=128, vocab_size=500, bos_token_id=0, eos_token_id=0)
        model = GPT2LMHeadModel(cfg)
        args.seq_len, args.core, args.band, args.eval_seqs = 64, 16, 8, 4
        g = torch.Generator().manual_seed(0)
        calib = torch.randint(0, 500, (8, 64), generator=g)
        ev = torch.randint(0, 500, (4, 64), generator=g)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        model = GPT2LMHeadModel.from_pretrained(args.model)
        calib, ev = load_ids(tok, args)
    model.to(dev).eval()
    blocks = model.transformer.h
    targets = [(i, name, getattr(blk.mlp, name)) for i, blk in enumerate(blocks) for name in ("c_fc", "c_proj")]

    # ---- calibration: input second moments of each routed matrix
    C = {}
    hooks = []
    for i, name, conv in targets:
        def hook(mod, inp, out, key=(i, name)):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double()
            if key not in C:
                C[key] = [torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=dev), 0]
            C[key][0] += x.T @ x
            C[key][1] += x.shape[0]
        hooks.append(conv.register_forward_hook(hook))
    with torch.no_grad():
        for row in calib:
            model(row[None].to(dev))
    for h in hooks:
        h.remove()

    # ---- decompose and patch
    ctrl = Controller()
    recon = []
    for n, (i, name, conv) in enumerate(targets):
        Cm = C[(i, name)][0] / C[(i, name)][1]
        b = Banded(conv.weight.data.T, Cm, args.core, args.band, args.m, args.outlier_frac,
                   args.ridge, args.seed + n, dev)
        recon.append(b.recon_rel_err)
        setattr(blocks[i].mlp, name, RoutedConv1D(conv, b, ctrl))
        C[(i, name)] = None
    routed = [getattr(blocks[i].mlp, name).banded for i, name, _ in targets]
    D = sum(b.dense_cost for b in routed)
    K = routed[0].K
    print(f"decomposed {len(routed)} matrices, ranks {routed[0].ranks}, max recon rel err {max(recon):.2e}, "
          f"{time.time() - t0:.0f}s")

    # ---- runs
    @torch.no_grad()
    def run(policy_for_seq):
        logits_all, cost_all, ks_all = [], [], []
        for s, row in enumerate(ev):
            ids = row[None].to(dev)
            pol = policy_for_seq(s)
            if pol is None:
                ctrl.policy = None
                lg = model(ids).logits[0]
                cost = torch.full((ids.shape[1],), D, dtype=torch.float64, device=dev)
            else:
                ctrl.start(pol, ids.shape[1], dev)
                lg = model(ids).logits[0]
                cost = ctrl.cost.clone()
                ks_all.append(list(ctrl.ks))
                ctrl.policy = None
            logits_all.append(lg.float().cpu())
            cost_all.append(cost.cpu())
        return torch.stack(logits_all), torch.stack(cost_all), ks_all

    dense_logits, _, _ = run(lambda s: None)
    dense_top = dense_logits.argmax(-1)
    logp_dense = torch.log_softmax(dense_logits, -1)
    labels = ev[:, 1:]

    def metrics(lg, cost):
        logq = torch.log_softmax(lg, -1)
        agree = float((lg.argmax(-1) == dense_top).float().mean())
        kl = float((logp_dense.exp() * (logp_dense - logq)).sum(-1).mean())
        nll = float(-logq[:, :-1].gather(-1, labels[..., None]).mean())
        return {"cost": float(cost.mean() / D), "agree": agree, "kl": kl, "ppl": math.exp(nll)}

    dense_nll = float(-logp_dense[:, :-1].gather(-1, labels[..., None]).mean())
    curves = {k: [] for k in ["fixed", "local_sketch", "local_exact", "random_matched", "global_margin"]}

    for k in range(K + 1):
        lg, c, _ = run(lambda s, k=k: Fixed(k))
        curves["fixed"].append({"k": k, **metrics(lg, c)})
        print("fixed", k, curves["fixed"][-1])

    gen = torch.Generator().manual_seed(args.seed)
    for eps in np.logspace(-2.5, 0, args.n_eps):
        for exact in (False, True):
            lg, c, ks = run(lambda s, e=eps, x=exact: Local(float(e), x))
            name = "local_exact" if exact else "local_sketch"
            mk = float(np.mean([float(k.float().mean()) for seq in ks for k in seq]))
            curves[name].append({"eps": float(eps), "mean_k": mk, **metrics(lg, c)})
            print(name, f"{eps:.4f}", curves[name][-1])
            if not exact:
                perm = [[k[torch.randperm(k.numel(), generator=gen).to(k.device)] for k in seq] for seq in ks]
                lr, cr, _ = run(lambda s, p=perm: Replay(p[s]))
                curves["random_matched"].append({"eps": float(eps), **metrics(lr, cr)})
                assert abs(float(cr.mean()) - float(c.mean())) < 1e-6 * D, "random_matched bill differs"

    for k_min in (0, 1, 2):
        lg1, c1, _ = run(lambda s, k=k_min: Fixed(k))
        top2 = lg1.topk(2, -1).values
        marg = top2[..., 0] - top2[..., 1]
        for q in np.linspace(0, 0.95, 20):
            tau = float(torch.quantile(marg.flatten(), float(q))) if q > 0 else -1.0
            flag = marg < tau
            lg = torch.where(flag[..., None], dense_logits, lg1)
            c = c1 + flag.double() * D
            curves["global_margin"].append({"k_min": k_min, "tau": tau, "flagged": float(flag.float().mean()),
                                            **metrics(lg, c)})

    # ---- matched-budget table
    def pareto(pts):
        pts = sorted(pts, key=lambda t: (t[0], -t[1]))
        out, best = [], -1
        for c, a in pts:
            if a > best:
                out.append((c, a))
                best = a
        return out

    def at(pts, b):
        f = pareto(pts)
        if not f or b < f[0][0]:
            return float("nan")
        if b >= f[-1][0]:
            return f[-1][1]
        for (c0, a0), (c1, a1) in zip(f, f[1:]):
            if c0 <= b <= c1:
                return a0 + (a1 - a0) * (b - c0) / (c1 - c0)
        return float("nan")

    table = {n: {str(b): at([(p["cost"], p["agree"]) for p in pts], b) for b in BUDGETS} for n, pts in curves.items()}

    # ---- preregistered verdict
    def pt(x):
        return 100 * x
    wins, never_worse, rand_ok = 0, True, 0
    for b in BUDGETS:
        L_ = table["local_sketch"][str(b)]
        best_att = np.nanmax([table["fixed"][str(b)], table["global_margin"][str(b)]])
        if L_ != L_:
            continue
        if pt(L_ - best_att) >= 1.0:
            wins += 1
            if pt(L_ - table["random_matched"][str(b)]) >= 1.0:
                rand_ok += 1
        for att in ("fixed", "global_margin"):
            v = table[att][str(b)]
            if v == v and pt(v - L_) > 1.0:
                never_worse = False
    kill = all(table["local_sketch"][str(b)] != table["local_sketch"][str(b)] or
               pt(table["local_sketch"][str(b)] - table["fixed"][str(b)]) < 1.0 for b in BUDGETS)
    verdict = "PASS" if (wins >= 2 and never_worse and rand_ok >= 2) else ("KILL" if kill else "NEITHER")

    receipt = {"prereg": "docs/PREREG.md", "args": vars(args), "ranks": routed[0].ranks,
               "max_recon_rel_err": max(recon), "dense_ppl": math.exp(dense_nll),
               "sketch_cost_frac": sum(b.sketch_cost for b in routed) / D,
               "core_cost_frac": sum(b.core_cost for b in routed) / D,
               "table_top1_agreement": table, "curves": curves, "verdict": verdict,
               "verdict_detail": {"budgets_won_by_1pt": wins, "never_worse_than_attackers_by_1pt": never_worse,
                                  "random_matched_lost_by_1pt_at_won_budgets": rand_ok},
               "notes": ["global_margin uses the true dense logits for flagged tokens and pays dense price once: "
                         "more generous than the prereg's rerun description",
                         "costs are analytic multiply-adds over the routed MLP matrices only; wall-clock not measured"],
               "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(receipt, open(args.out, "w"), indent=1)

    print(f"\ndense ppl {receipt['dense_ppl']:.3f}; core {receipt['core_cost_frac']:.3f}, "
          f"sketch {receipt['sketch_cost_frac']:.3f} of dense MLP cost")
    print("router            " + "".join(f"  b={b:<4}" for b in BUDGETS))
    for n, row in table.items():
        print(f"{n:<17} " + "".join("   nan " if row[str(b)] != row[str(b)] else f" {100 * row[str(b)]:6.2f}"
                                    for b in BUDGETS))
    print(f"\nVERDICT: {verdict}  {receipt['verdict_detail']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
