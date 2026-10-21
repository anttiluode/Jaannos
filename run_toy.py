"""Toy gates G0-G3 from docs/PREREG.md. CPU, numpy only, about a minute.

    python run_toy.py            # writes results/toy_receipt.json and prints the tables
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from jaannos.bands import BandedLinear, tails_from_band_norms
from jaannos.routing import (Fixed, Local, Replay, ToyNet, at_budget, global_margin,
                             margin, permute_k, score)

BUDGETS = [0.5, 0.6, 0.7, 0.8]
EPS = list(np.round(np.logspace(-3, 0, 25), 6))
N = 4000
SEED = 20261021
# v0 used alpha = 1.0 and was uninformative (core-only already 99.7% agreement, ceiling);
# its receipt is kept as results/toy_receipt_v0_ceiling.json. v1 flattens the spectrum.
ALPHA = 0.35


def world_null(net, rng):
    return rng.standard_normal((N, net.d))


def world_constructed(net, rng, p=0.2, amp=20.0):
    """Base inputs live in layer 0's top input directions; a fraction p also carries energy
    along its tail input directions (bands 5-8). Built so a local residue test CAN see it."""
    B = net.blin[0].B
    Vtop = B[:32] / np.linalg.norm(B[:32], axis=1, keepdims=True)
    Vtail = B[80:] / np.linalg.norm(B[80:], axis=1, keepdims=True)
    X = rng.standard_normal((N, 32)) @ Vtop
    hard = rng.random(N) < p
    X[hard] += amp * rng.standard_normal((hard.sum(), Vtail.shape[0])) @ Vtail / np.sqrt(Vtail.shape[0])
    X += 0.05 * rng.standard_normal(X.shape)
    return X, hard


def evaluate(net, X):
    dense = net.dense(X)
    D = net.dense_cost_per_token
    curves = {k: [] for k in ["fixed", "local_sketch", "local_exact", "random_matched", "global_margin"]}
    accounting = {"recount_max_abs_err": 0.0, "random_vs_local_cost_max_abs_err": 0.0}

    for k in range(net.blin[0].K + 1):
        lg, c, _ = net.run(X, Fixed(k))
        a, kl = score(lg, dense)
        curves["fixed"].append({"k": k, "cost": float(c.mean() / D), "agree": a, "kl": kl})

    for eps in EPS:
        for exact in (False, True):
            lg, c, ks = net.run(X, Local(eps, exact=exact))
            a, kl = score(lg, dense)
            name = "local_exact" if exact else "local_sketch"
            curves[name].append({"eps": eps, "cost": float(c.mean() / D), "agree": a, "kl": kl,
                                 "mean_k": float(np.mean([kk.mean() for kk in ks]))})
            if not exact:
                recount = sum(net.blin[i].token_cost(kk, sketch=True) for i, kk in enumerate(ks))
                accounting["recount_max_abs_err"] = max(accounting["recount_max_abs_err"],
                                                        float(np.abs(recount - c).max()))
                for s in range(3):
                    lr, cr, _ = net.run(X, Replay(permute_k(ks, SEED + s)))
                    ar, klr = score(lr, dense)
                    accounting["random_vs_local_cost_max_abs_err"] = max(
                        accounting["random_vs_local_cost_max_abs_err"], float(abs(cr.mean() - c.mean())))
                    curves["random_matched"].append({"eps": eps, "seed": s, "cost": float(cr.mean() / D),
                                                     "agree": ar, "kl": klr})

    for k_min in (0, 1, 2):
        m1 = margin(net.run(X, Fixed(k_min))[0])
        for q in np.linspace(0, 0.95, 20):
            tau = float(np.quantile(m1, q)) if q > 0 else -1.0
            lg, c, frac = global_margin(net, X, dense, k_min, tau)
            a, kl = score(lg, dense)
            curves["global_margin"].append({"k_min": k_min, "tau": tau, "flagged": frac,
                                            "cost": float(c.mean() / D), "agree": a, "kl": kl})

    table = {}
    for name, pts in curves.items():
        pp = [(p["cost"], p["agree"]) for p in pts]
        if name == "random_matched":   # average the permutation seeds at each eps first
            by = {}
            for p in pts:
                by.setdefault(p["eps"], []).append((p["cost"], p["agree"]))
            pp = [(np.mean([c for c, _ in v]), np.mean([a for _, a in v])) for v in by.values()]
        table[name] = {str(b): at_budget(pp, b) for b in BUDGETS}
    return curves, table, accounting


def gate0():
    out = {}
    net = ToyNet(alpha=ALPHA, seed=SEED)
    rel = max(np.linalg.norm(b.dense_equivalent() - W) / np.linalg.norm(W) for b, W in zip(net.blin, net.W))
    out["reconstruction_max_rel_frob"] = float(rel)
    X = np.random.default_rng(1).standard_normal((256, net.d))
    lg, _, _ = net.run(X, Fixed(net.blin[0].K))
    out["all_bands_vs_dense_max_abs"] = float(np.abs(lg - net.dense(X)).max())

    rng = np.random.default_rng(2)
    errs = {}
    for shape in [(3072, 768), (768, 3072)]:
        W = rng.standard_normal(shape) / np.sqrt(shape[1])
        b = BandedLinear(W, [256] + [64] * 8, m=4, seed=3)
        X = rng.standard_normal((2000, shape[1]))
        est = tails_from_band_norms(b.band_norms_sketch(X))[:, 0]
        tru = tails_from_band_norms(b.band_norms_exact(X))[:, 0]
        errs[f"{shape[0]}x{shape[1]}"] = float(np.median(np.abs(est / tru - 1)))
        errs[f"{shape[0]}x{shape[1]}_sketch_cost_frac_of_dense"] = b.sketch_cost / b.dense_cost
    out["sketch_median_rel_err_tail0"] = errs
    out["pass"] = bool(rel < 1e-10 and out["all_bands_vs_dense_max_abs"] < 1e-9
                       and all(v < 0.15 for k, v in errs.items() if not k.endswith("dense")))
    return out


def fmt(x):
    return "   nan" if x != x else f"{100 * x:6.2f}"


def print_table(title, table):
    print(f"\n{title}  (top-1 agreement with dense, %, at matched cost)")
    print("router            " + "".join(f"  b={b:<4}" for b in BUDGETS))
    for name, row in table.items():
        print(f"{name:<17} " + "".join(f" {fmt(row[str(b)])}" for b in BUDGETS))


def main():
    os.makedirs("results", exist_ok=True)
    receipt = {"prereg": "docs/PREREG.md", "seed": SEED, "N": N, "budgets": BUDGETS}

    g0 = gate0()
    receipt["G0"] = g0
    print("G0 identity:", "PASS" if g0["pass"] else "FAIL", json.dumps(g0, indent=1))
    if not g0["pass"]:
        json.dump(receipt, open("results/toy_receipt.json", "w"), indent=1)
        sys.exit(1)

    net = ToyNet(alpha=ALPHA, seed=SEED)
    rng = np.random.default_rng(SEED)
    receipt["toy_config"] = {"d": net.d, "d_ff": net.d_ff, "L": net.L, "ranks": net.ranks,
                             "m_per_band": 3,
                             "core_cost_frac": float(sum(b.core_cost for b in net.blin) / net.dense_cost_per_token),
                             "band_cost_frac_each": float(sum(b.band_cost[0] for b in net.blin) / net.dense_cost_per_token),
                             "sketch_cost_frac": float(sum(b.sketch_cost for b in net.blin) / net.dense_cost_per_token)}

    # G2 null world
    Xn = world_null(net, rng)
    cn, tn, accn = evaluate(net, Xn)
    gap = [tn["local_sketch"][str(b)] - tn["fixed"][str(b)] for b in BUDGETS]
    gap = [g for g in gap if g == g]
    g2 = {"local_minus_fixed_pts": [100 * g for g in gap], "pass": bool(all(100 * g <= 1.0 for g in gap))}
    receipt["G2_null"] = {"table": tn, "curves": cn, "accounting": accn, "verdict": g2}
    print_table("G2 null world", tn)
    print("G2 verdict:", "PASS (no free lunch)" if g2["pass"] else "FAIL (local beats fixed on noise -> leak)", g2)

    # G3 constructed world
    Xc, hard = world_constructed(net, rng)
    cc, tc, accc = evaluate(net, Xc)
    core_lg = net.run(Xc, Fixed(0))[0]
    dense_c = net.dense(Xc)
    dis = core_lg.argmax(1) != dense_c.argmax(1)
    receipt["G3_constructed"] = {"table": tc, "curves": cc, "accounting": accc,
                                 "hard_frac": float(hard.mean()),
                                 "core_only_disagree_rate_hard": float(dis[hard].mean()),
                                 "core_only_disagree_rate_easy": float(dis[~hard].mean()),
                                 "note": "mechanism check only, built for the local test; not evidence"}
    # direct mechanism diagnostics: does the sketch rank tokens like the exact tail, and does the
    # local router give the designed-hard tokens more rank at the matrix where they are hard?
    b0 = net.blin[0]
    est = tails_from_band_norms(b0.band_norms_sketch(Xc))[:, 0]
    tru = tails_from_band_norms(b0.band_norms_exact(Xc))[:, 0]
    rk = lambda v: np.argsort(np.argsort(v))
    spearman = float(np.corrcoef(rk(est), rk(tru))[0, 1])
    mid = EPS[len(EPS) // 2]
    _, _, ks = net.run(Xc, Local(mid))
    diag = {"layer0_sketch_vs_exact_tail_spearman": spearman, "eps": mid,
            "layer0_mean_k_hard": float(ks[0][hard].mean()), "layer0_mean_k_easy": float(ks[0][~hard].mean()),
            "all_layers_mean_k_hard": float(np.mean([k[hard].mean() for k in ks])),
            "all_layers_mean_k_easy": float(np.mean([k[~hard].mean() for k in ks]))}
    receipt["G3_constructed"]["diagnostics"] = diag
    print_table("G3 constructed world (mechanism check, NOT evidence)", tc)
    print("G3 diagnostics:", json.dumps(diag))
    print(f"core-only disagreement: hard {dis[hard].mean():.3f}, easy {dis[~hard].mean():.3f}")

    # G1 accounting, both worlds
    g1_err = max(accn["recount_max_abs_err"], accc["recount_max_abs_err"])
    g1_rand = max(accn["random_vs_local_cost_max_abs_err"], accc["random_vs_local_cost_max_abs_err"])
    receipt["G1"] = {"recount_max_abs_err": g1_err, "random_vs_local_cost_max_abs_err": g1_rand,
                     "pass": bool(g1_err < 1e-6 and g1_rand < 1e-6)}
    print("G1 accounting:", "PASS" if receipt["G1"]["pass"] else "FAIL", receipt["G1"])

    json.dump(receipt, open("results/toy_receipt.json", "w"), indent=1, default=float)
    print("\nwrote results/toy_receipt.json")


if __name__ == "__main__":
    main()
