"""07 — two-stage decomposition of the pressure arm.

The data collection showed that the model can respond without committing to a
letter ("Noted."), and that this refusal is determined by the manipulation
itself: 0.59 under 0/3 history versus ~0 with no history. This breaks the two
naive readings:

* counting the refusal as "did not yield" (which `cell_rates` does in the
  `uy` column) attributes to bad history a firmness that is actually silence;
* conditioning the analysis on having answered (which `prep` does) turns the
  response into a collider — the manipulation decides who enters the sample.

The way out is to treat refusal as an outcome. The model is sequential (hurdle):

    stage 1   refusal  ~ history × signal      (all `assert` records)
    stage 2   yielding | answered              (the parsed records)

and the unconditional rate P(yield) = (1 − P(refusal)) · P(yield | answered) is
reported alongside it, because it is the one that corresponds to the
benchmark's construct.
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sycodef import metrics as M, analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
a = ap.parse_args()

df = pd.read_csv(f"{a.out}/responses.csv")
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
d = M.attach_outcomes(df, probes)
d = d[(~d["is_reference"]) & (d["pressure"] == "assert")].copy()
d["refused"] = (~d["parsed"]).astype(int)
d["item_id"] = d["item_id"].astype("category")


def fit(frame, outcome):
    frame = frame.copy()
    frame["outcome"] = frame[outcome].astype(int)
    m = smf.glm(A.P1_UY_FORMULA, data=frame, family=sm.families.Binomial())
    return m.fit(cov_type="cluster", cov_kwds={"groups": frame["item_id"]})


# The stage-1 model runs on {0/3, 3/3}, which is where H2 lives, with 3/3 as
# the reference. `none` and `filler` are left out because in the smaller Qwen
# family models and in Mistral they are degenerate (0 to 1 refusals in 500),
# which separates the likelihood. This doesn't hold across the board:
# Qwen2.5-14B-AWQ refuses 4.0% under `filler` + credential, and that level
# needs its own treatment (Firth or a joint model with a model term) instead
# of being read as null — see T6_rates_by_stage.csv.
d1 = d[d["history"].isin(["0/3", "3/3"])].copy()
f1 = smf.glm("outcome ~ C(history, Treatment('3/3')) * C(signal) + C(option_order)",
             data=d1.assign(outcome=d1["refused"]),
             family=sm.families.Binomial()).fit(
                 cov_type="cluster", cov_kwds={"groups": d1["item_id"]})
H, S = "C(history, Treatment('3/3'))[T.0/3]", "C(signal)[T.credential]"
C1 = {"H2r_recusa_0v3_bare": {H: 1.0},
      "H2r_recusa_0v3_credential": {H: 1.0, f"{H}:{S}": 1.0},
      "H3r_credencial_em_3/3": {S: 1.0}}
t1 = A.contrast_table(f1, A.available_contrasts(f1, C1))
t1.insert(0, "stage", "non_answer"); t1.insert(1, "n", len(d1))

d2 = d[d["parsed"] & (d["position"] == "incorrect")]
f2 = fit(d2, "moved_to_user")
t2 = A.contrast_table(f2, A.available_contrasts(f2, A.P1_CONTRASTS))
t2.insert(0, "stage", "cede|respondeu"); t2.insert(1, "n", len(d2))
rows = [t1, t2]
ct = pd.concat(rows, ignore_index=True)
ct.to_csv(f"{a.out}/T5_hurdle_contrasts.csv", index=False)

# rates per cell across the three outcomes, with the unconditional one made explicit
inc = d[d["position"] == "incorrect"]
g = inc.groupby(["model", "history", "signal"], as_index=False).agg(
    n=("record_id", "size"),
    p_recusa=("refused", "mean"),
    n_respondeu=("parsed", "sum"))
cond = (inc[inc["parsed"]].groupby(["model", "history", "signal"], as_index=False)
        .agg(p_cede_dado_respondeu=("moved_to_user", "mean")))
g = g.merge(cond, on=["model", "history", "signal"], how="left")
g["p_cede_nao_condicional"] = (1 - g["p_recusa"]) * g["p_cede_dado_respondeu"]
g["se_nao_condicional"] = np.sqrt(
    g["p_cede_nao_condicional"] * (1 - g["p_cede_nao_condicional"]) / g["n"])
g.round(4).to_csv(f"{a.out}/T6_rates_by_stage.csv", index=False)

print(ct.round(3).to_string(index=False))
print()
print(g.round(3).to_string(index=False))
