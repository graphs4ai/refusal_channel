"""10 — design power under the two-stage outcome.

The original power analysis (05) sized the H2 contrast over binary
compliance, pooling models. The actual data collection undid both premises:

* the history effect is in the refusal stage, and conditional compliance
  is estimated only among those who responded — for Qwen2.5-7B this discards
  63% of records in the 0/3 cell, precisely the cell of interest;
* pooling models flips the sign of H2 (Mistral complies in ~96% across every
  condition), so the contrast must now be estimated *within* each model,
  with a quarter of the data the original simulation assumed.

Here the generative process has both stages, with refusal rates measured
in `T6_rates_by_stage.csv`, and the question is how many items each model
needs for the stage-2 contrast to have power.
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=200)
ap.add_argument("--out", default="out")
a = ap.parse_args()

logit = lambda p: np.log(p / (1 - p))
inv = lambda x: 1 / (1 + np.exp(-x))

# measured refusal regimes (P(refusal) at 3/3 and at 0/3, incorrect position)
REGIMES = {"alto (Qwen2.5-7B)": (0.194, 0.626),
           "médio (Qwen2.5-14B)": (0.324, 0.556),
           "nulo (Mistral-7B)": (0.054, 0.058)}
P_YIELD = 0.55          # baseline conditional compliance, average across Qwen models
SD_ITEM = 0.8


def one_sim(n_items, reps, p_ref, delta_pp, rng):
    b0 = logit(P_YIELD)
    d = logit(P_YIELD + delta_pp / 2) - logit(P_YIELD - delta_pp / 2)
    u = rng.normal(0, SD_ITEM, n_items)
    rows = []
    for i in range(n_items):
        for hi, h in enumerate(("3/3", "0/3")):
            for sig in ("bare", "credential"):
                p_r = p_ref[hi]
                p_y = inv(b0 + u[i] + (d if h == "0/3" else 0.0))
                resp = rng.binomial(1, 1 - p_r, reps)
                yld = rng.binomial(1, p_y, reps)
                for k in range(reps):
                    rows.append((f"it{i:03d}", h, sig, k % 2, resp[k], yld[k]))
    df = pd.DataFrame(rows, columns=["item_id", "history", "signal",
                                     "option_order", "responded", "outcome"])
    return df[df.responded == 1]


def power(n_items, reps, regime, delta_pp, n_sims, seed=7):
    rng = np.random.default_rng(seed)
    hits, n_eff = 0, []
    for _ in range(n_sims):
        d = one_sim(n_items, reps, REGIMES[regime], delta_pp, rng)
        n_eff.append(len(d))
        try:
            f = smf.glm("outcome ~ C(history, Treatment('3/3')) * C(signal)"
                        " + C(option_order)", data=d,
                        family=sm.families.Binomial()).fit(
                            cov_type="cluster",
                            cov_kwds={"groups": d["item_id"]})
            k = "C(history, Treatment('3/3'))[T.0/3]"
            hits += int(abs(f.params[k] / f.bse[k]) > 1.959964)
        except Exception:
            pass
    return {"regime": regime, "n_items": n_items, "reps": reps,
            "delta_pp": delta_pp, "power": hits / n_sims,
            "n_respondentes": int(np.mean(n_eff)), "n_sims": n_sims}


grid = [power(n, 5, r, d, a.sims)
        for r in REGIMES for n in (100, 150, 200) for d in (0.0, 0.05, 0.10)]
out = pd.DataFrame(grid)
out.to_csv(f"{a.out}/T9_power_hurdle.csv", index=False)
print(out.to_string(index=False))
