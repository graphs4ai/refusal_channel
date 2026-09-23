"""08 — P1 contrasts by model.

Pooling the three models into a single regression without a model term flips the sign of
H2: Mistral yields in ~96% of records under every condition, and the sample
composition carries the coefficient. The contrast is estimated within each model.
"""
import argparse
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sycodef import metrics as M, analysis as A

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out")
a = ap.parse_args()
d0 = M.attach_outcomes(pd.read_csv(f"{a.out}/responses.csv"),
                       pd.read_csv(f"{a.out}/item_pool_probes.csv"))
F = "outcome ~ C(history, Treatment('3/3')) * C(signal) + C(option_order)"
H, S = "C(history, Treatment('3/3'))[T.0/3]", "C(signal)[T.credential]"
CS = {"recusa_0v3": {H: 1.0}, "credencial": {S: 1.0}}

rows = []
for m, dm in d0[(~d0.is_reference) & (d0.pressure == "assert")].groupby("model"):
    sub = dm[dm.history.isin(["0/3", "3/3"])]
    for stage, frame, outcome in [
            ("non_answer", sub, (~sub["parsed"]).astype(int)),
            ("cede|respondeu", sub[sub.parsed & (sub.position == "incorrect")], None)]:
        fr = frame.copy()
        fr["outcome"] = (outcome if outcome is not None
                         else fr["moved_to_user"]).astype(int)
        if fr["outcome"].nunique() < 2:
            rows.append(dict(model=m, stage=stage, contrast="recusa_0v3",
                             n=len(fr), note="no variation - not estimable"))
            continue
        fit = smf.glm(F, data=fr, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": fr["item_id"].astype("category")})
        t = A.contrast_table(fit, A.available_contrasts(fit, CS))
        # cell with no events => complete separation: the coefficient is an artifact
        # of the optimizer (|log-odds| in the tens), not an estimate.
        cells = fr.groupby(["history", "signal"])["outcome"].mean()
        if ((cells == 0) | (cells == 1)).any():
            sep = t["log_odds"].abs() > 10
            t.loc[sep, ["log_odds", "se", "z", "p", "or", "or_lo", "or_hi",
                        "p_holm"]] = float("nan")
            t["note"] = ["complete separation - not estimable" if x else ""
                         for x in sep]
        t.insert(0, "model", m); t.insert(1, "stage", stage); t.insert(2, "n", len(fr))
        rows.append(t)
out = pd.concat([r if isinstance(r, pd.DataFrame) else pd.DataFrame([r])
                 for r in rows], ignore_index=True)
out.to_csv(f"{a.out}/T7_contrasts_by_model.csv", index=False)
print(out.round(3).to_string(index=False))
