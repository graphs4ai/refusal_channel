"""Inference.

Pre-register two distinct things and do not confuse them:

1. Primary analysis — logistic regression with item-cluster-robust
   standard errors (`fit_logit_cluster`). This is what supports the
   contrasts for H1-H4: it does not depend on variational approximation,
   accepts full interactions, and the item cluster is the unit of
   replication of the design.
2. Confirmatory analysis — logistic GLMM with crossed random intercepts
   for item and model (`fit_glmm`, statsmodels variational Bayes). Serves
   to (a) show that the contrasts survive crossed random effects and
   (b) provide the variance components that P2 needs.

P2's ranking is compared via Kendall tau with an item permutation test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# H2/H3 are about unjustified deference: fit on the subset
# position == "incorrect", where the contrasts are main effects.
P1_UY_FORMULA = ("outcome ~ C(history, Treatment('none')) * C(signal) "
                 "+ C(option_order)")
# H4 (discrimination) needs the interaction with user position.
P1_FULL_FORMULA = ("outcome ~ C(history, Treatment('none')) * C(position) "
                   "* C(signal) + C(option_order)")
P2_FORMULA = "outcome ~ C(pressure, Treatment('none'))"


def prep(d: pd.DataFrame, outcome: str, subset=None) -> pd.DataFrame:
    x = d[~d["is_reference"]].copy()
    if subset is not None:
        x = x[subset(x)]
    x = x[x["parsed"]]
    x["outcome"] = x[outcome].astype(int)
    x["item_id"] = x["item_id"].astype("category")
    return x


def fit_logit_cluster(d: pd.DataFrame, formula: str = P1_UY_FORMULA,
                      cluster: str = "item_id"):
    """Logistic regression with item-cluster-robust SE. Primary analysis."""
    m = smf.glm(formula, data=d, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})


def fit_glmm(d: pd.DataFrame, formula: str = P1_UY_FORMULA,
             groups=("item_id", "model"), vcp_p: float = 2.0):
    """Logistic GLMM with crossed random intercepts. Confirmatory."""
    vc = {g: f"0 + C({g})" for g in groups}
    mod = BinomialBayesMixedGLM.from_formula(formula, vc, d, vcp_p=vcp_p)
    return mod.fit_vb(verbose=False)


def variance_components(d: pd.DataFrame,
                        groups=("item_id", "model", "cell_id")) -> pd.DataFrame:
    """Variance components of the flip (P2). Reports the posterior SD of each
    factor on the logit scale and the fraction of variance explained, adding
    the residual logistic variance (pi^2/3) as a level-1 component."""
    x = d[~d["is_reference"]].copy()
    x["outcome"] = x["flip"].astype(int)
    vc = {g: f"0 + C({g})" for g in groups}
    fit = BinomialBayesMixedGLM.from_formula("outcome ~ 1", vc, x).fit_vb(verbose=False)
    sds = np.exp(np.asarray(fit.vcp_mean))
    var = sds ** 2
    resid = np.pi ** 2 / 3
    total = var.sum() + resid
    rows = [{"component": g, "sd_logit": float(s), "var": float(v),
             "share": float(v / total)} for g, s, v in zip(groups, sds, var)]
    rows.append({"component": "residual(logistic)", "sd_logit": float(np.sqrt(resid)),
                 "var": float(resid), "share": float(resid / total)})
    rows.append({"component": "ICC_item", "sd_logit": np.nan, "var": np.nan,
                 "share": float(var[0] / total)})
    return pd.DataFrame(rows)


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def contrast_table(fit, specs: dict) -> pd.DataFrame:
    """`specs`: name -> dict {term: weight}. Wald test on the linear
    combination, with Holm correction across the contrast family."""
    names = list(fit.params.index)
    rows = []
    for label, weights in specs.items():
        L = np.zeros(len(names))
        for term, w in weights.items():
            if term not in names:
                raise KeyError(f"term missing from model: {term}")
            L[names.index(term)] = w
        est = float(L @ fit.params.values)
        se = float(np.sqrt(L @ fit.cov_params().values @ L))
        z = est / se if se > 0 else np.nan
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(z))))
        rows.append({"contrast": label, "log_odds": est, "se": se, "z": z,
                     "p": p, "or": float(np.exp(est)),
                     "or_lo": float(np.exp(est - 1.96 * se)),
                     "or_hi": float(np.exp(est + 1.96 * se))})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].values)
    return out


def permutation_ranking_test(d: pd.DataFrame, B: int = 2000,
                             seed: int = 20261012) -> dict:
    """P2's H2: is the difference between the raw ranking and the
    null-corrected ranking larger than expected from item resampling?"""
    from scipy.stats import kendalltau
    from .metrics import indices
    rng = np.random.default_rng(seed)
    obs = indices(d)
    tau_obs = kendalltau(obs["flip_raw"].rank(), obs["excess_flip"].rank())[0]
    items = d["item_id"].unique()
    by_item = {i: g for i, g in d.groupby("item_id")}
    taus = []
    for _ in range(B):
        samp = rng.choice(items, size=len(items), replace=True)
        boot = pd.concat([by_item[i] for i in samp], ignore_index=True)
        bi = indices(boot)
        taus.append(kendalltau(bi["flip_raw"].rank(), bi["excess_flip"].rank())[0])
    taus = np.asarray(taus, float)
    return {"tau_observed": float(tau_obs),
            "tau_boot_mean": float(np.nanmean(taus)),
            "tau_boot_lo": float(np.nanquantile(taus, 0.025)),
            "tau_boot_hi": float(np.nanquantile(taus, 0.975)),
            "p_tau_ge_1": float(np.nanmean(taus >= 0.999))}


_H = "C(history, Treatment('none'))"
_S = "C(signal)[T.credential]"

# Pre-registered contrasts, fitted on position == "incorrect"
# (unjustified deference) with P1_UY_FORMULA.
P1_CONTRASTS = {
    # H1 is the null hypothesis of H2: invariance to history
    # H2: bad history (0/3) yields more than good history (3/3), bare signal
    "H2_delta_0v3_bare": {f"{_H}[T.0/3]": 1.0, f"{_H}[T.3/3]": -1.0},
    # H2 in the presence of a credential (does the history effect survive the label?)
    "H2_delta_0v3_credential": {f"{_H}[T.0/3]": 1.0, f"{_H}[T.3/3]": -1.0,
                                f"{_H}[T.0/3]:{_S}": 1.0, f"{_H}[T.3/3]:{_S}": -1.0},
    # H3: effect of the stated credential minus the effect of the demonstrated history
    "H3_cred_minus_hist": {_S: 1.0, f"{_H}[T.0/3]": -1.0, f"{_H}[T.3/3]": 1.0},
    # isolated credential effect, without history
    "H3_credential_only": {_S: 1.0},
    # length/exposure control: filler should not differ from none
    "CTRL_filler_vs_none": {f"{_H}[T.filler]": 1.0},
    # monotonicity (only in the ideal design, with 4 accuracy levels):
    # linear contrast in k = 0,1,2,3 correct
    "H2_linear_trend": {f"{_H}[T.0/3]": 1.5, f"{_H}[T.1/3]": 0.5,
                        f"{_H}[T.2/3]": -0.5, f"{_H}[T.3/3]": -1.5},
}

# H4 (discrimination): does history improve the separation between correct
# and incorrect user, or does it just shift global compliance? Fit with
# P1_FULL_FORMULA.
P1_H4_CONTRAST = {
    "H4_history_x_position": {
        f"{_H}[T.0/3]:C(position)[T.incorrect]": 1.0,
        f"{_H}[T.3/3]:C(position)[T.incorrect]": -1.0},
}


def available_contrasts(fit, specs: dict) -> dict:
    """Filters specs whose terms exist in the fitted model (the minimal
    design does not have the 1/3 and 2/3 levels, so it does not have the
    linear contrast)."""
    names = set(fit.params.index)
    return {k: v for k, v in specs.items() if set(v).issubset(names)}
