"""Data-generating process and power analysis.

Serves three functions in the plan:
  (a) validate that the estimators in `metrics.py` recover known parameters
      before spending tokens on the API;
  (b) give the sample size needed per cell for the primary contrast
      (reliability delta) and for the credential-vs-history difference;
  (c) provide the power curve that goes into the paper's appendix.

The DGP has random item and model intercepts on the logit scale, because that
is the structure that determines power: with 100 items, what limits power is
not the number of generations, it's the between-item variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def inv_logit(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_yield(n_items=100, n_models=6, reps=3, option_orders=2,
                   p_yield_base=0.35,
                   delta_pp=0.08, cred_pp=0.12, sd_item=0.8, sd_model=0.5,
                   history_levels=("none", "0/3", "3/3", "filler"),
                   seed=0) -> pd.DataFrame:
    """Generates `moved_to_user` in the position=incorrect cell (the one that
    defines unjustified yielding), for the history x signal crossed design.

    `delta_pp`: effect of bad vs good history, in percentage points on the
    probability scale around `p_yield_base`.
    `cred_pp`: effect of declared credential, likewise.
    """
    rng = np.random.default_rng(seed)
    b0 = logit(p_yield_base)
    d_lo = logit(p_yield_base + delta_pp / 2) - b0
    d_hi = logit(p_yield_base - delta_pp / 2) - b0
    b_cred = logit(p_yield_base + cred_pp) - b0
    hist_eff = {"none": 0.0, "0/3": d_lo, "1/3": d_lo / 3, "2/3": d_hi / 3,
                "3/3": d_hi, "filler": 0.0}
    u_item = rng.normal(0, sd_item, n_items)
    u_model = rng.normal(0, sd_model, n_models)
    rows = []
    for i in range(n_items):
        for m in range(n_models):
            for h in history_levels:
                for sig in ("bare", "credential"):
                    eta = (b0 + u_item[i] + u_model[m] + hist_eff[h]
                           + (b_cred if sig == "credential" else 0.0))
                    p = inv_logit(eta)
                    # `reps` replicates per cell; paraphrase and option order
                    # vary *between* replicates (they are not crossed), as in
                    # design.build_design
                    y = rng.binomial(1, p, reps)
                    for k, yy in enumerate(y):
                        rows.append({"item_id": f"it{i:03d}", "model": f"m{m}",
                                     "history": h, "signal": sig,
                                     "position": "incorrect", "lang": "en",
                                     "option_order": k % option_orders,
                                     "outcome": int(yy), "p_true": p})
    return pd.DataFrame(rows)


FORMULA = "outcome ~ C(history, Treatment('none')) * C(signal) + C(option_order)"


def marginal_targets(df: pd.DataFrame) -> dict:
    """Targets on the **marginal** (population) scale, which is what is
    estimated by a logistic regression with cluster SEs.

    With random item and model intercepts, the marginal effect is attenuated
    relative to the conditional effect that generated the data (factor
    ~1/sqrt(1+0.346·σ²)). Comparing the marginal estimate to the conditional
    target produces an apparent bias of ~-20% that doesn't exist: it's a
    target mismatch, not an estimator error. The targets below come from the
    true average probabilities per cell, so they are exact for the simulated
    design.
    """
    g = df.groupby(["history", "signal"])["p_true"].mean()
    h2 = logit(g[("0/3", "bare")]) - logit(g[("3/3", "bare")])
    cred = logit(g[("none", "credential")]) - logit(g[("none", "bare")])
    return {"H2": float(h2), "H3": float(cred - h2), "credential": float(cred)}


def fit_once(df: pd.DataFrame):
    m = smf.glm(FORMULA, data=df, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": df["item_id"]})


def contrast(fit, weights: dict):
    names = list(fit.params.index)
    L = np.zeros(len(names))
    for t, w in weights.items():
        L[names.index(t)] = w
    est = float(L @ fit.params.values)
    se = float(np.sqrt(L @ fit.cov_params().values @ L))
    return est, se


H2 = {"C(history, Treatment('none'))[T.0/3]": 1.0,
      "C(history, Treatment('none'))[T.3/3]": -1.0}
H3 = {"C(signal)[T.credential]": 1.0,
      "C(history, Treatment('none'))[T.0/3]": -1.0,
      "C(history, Treatment('none'))[T.3/3]": 1.0}


def power_curve(deltas_pp=(0.0, 0.03, 0.05, 0.08, 0.12), n_sims=200,
                n_items=100, reps=3, n_models=6, alpha=0.05,
                cred_pp=0.12, sd_item=0.8, seed=1234) -> pd.DataFrame:
    """Power of the H2 contrast (and type I error at delta_pp=0), with the
    credential effect held fixed. Uses only the position=incorrect cell."""
    zcrit = 1.959964
    out = []
    for dpp in deltas_pp:
        sig, ests, tgts = 0, [], []
        for s in range(n_sims):
            df = simulate_yield(n_items=n_items, n_models=n_models, reps=reps,
                                delta_pp=dpp, cred_pp=cred_pp, sd_item=sd_item,
                                seed=seed + s)
            fit = fit_once(df)
            e, se = contrast(fit, H2)
            ests.append(e)
            tgts.append(marginal_targets(df)["H2"])
            sig += int(abs(e / se) > zcrit)
        out.append({"delta_pp": dpp, "n_items": n_items, "reps": reps,
                    "n_models": n_models, "power": sig / n_sims,
                    "mean_est_logodds": float(np.mean(ests)),
                    "marginal_target_logodds": float(np.mean(tgts)),
                    "bias": float(np.mean(ests) - np.mean(tgts)),
                    "n_rows_per_sim": int(len(df)), "n_sims": n_sims})
    return pd.DataFrame(out)


def recovery_check(n_sims=50, delta_pp=0.08, cred_pp=0.12, seed=99) -> pd.DataFrame:
    """Joint recovery of H2 and H3 against the marginal targets."""
    rows = []
    for s in range(n_sims):
        df = simulate_yield(delta_pp=delta_pp, cred_pp=cred_pp, seed=seed + s)
        fit = fit_once(df)
        e2, s2 = contrast(fit, H2)
        e3, s3 = contrast(fit, H3)
        t = marginal_targets(df)
        rows.append({"h2_est": e2, "h2_se": s2, "h2_t": t["H2"],
                     "h3_est": e3, "h3_se": s3, "h3_t": t["H3"]})
    r = pd.DataFrame(rows)
    res = []
    for lab, k in [("H2 (reliability delta)", "h2"),
                   ("H3 (credential - history)", "h3")]:
        res.append({"contrast": lab, "marginal_target": r[f"{k}_t"].mean(),
                    "mean_est": r[f"{k}_est"].mean(), "sd_est": r[f"{k}_est"].std(),
                    "mean_se": r[f"{k}_se"].mean(),
                    "coverage95": float(np.mean(
                        np.abs(r[f"{k}_est"] - r[f"{k}_t"]) < 1.96 * r[f"{k}_se"]))})
    return pd.DataFrame(res)
