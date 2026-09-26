"""04 — outcomes, indices, contrasts and variance components."""
import argparse, json
import pandas as pd
from sycodef import metrics as M, analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--bootstrap", type=int, default=1000)
ap.add_argument("--permutations", type=int, default=2000)
ap.add_argument("--out", default="out")
a = ap.parse_args()

df = pd.read_csv(f"{a.out}/responses.csv")
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
d = M.attach_outcomes(df, probes)
d.to_csv(f"{a.out}/records_with_outcomes.csv", index=False)
M.cell_rates(d).to_csv(f"{a.out}/T1_cell_rates.csv", index=False)

idx = M.bootstrap_indices(d, B=a.bootstrap) if a.bootstrap else M.indices(d)
idx.to_csv(f"{a.out}/indices.csv", index=False)

uy = A.prep(d, "moved_to_user",
            subset=lambda x: (x["pressure"] == "assert") & (x["position"] == "incorrect"))
fit = A.fit_logit_cluster(uy, formula=A.P1_UY_FORMULA)
ct = A.contrast_table(fit, A.available_contrasts(fit, A.P1_CONTRASTS))
full = A.prep(d, "moved_to_user", subset=lambda x: x["pressure"] == "assert")
fit4 = A.fit_logit_cluster(full, formula=A.P1_FULL_FORMULA)
ct = pd.concat([ct, A.contrast_table(fit4, A.available_contrasts(fit4, A.P1_H4_CONTRAST))],
               ignore_index=True)
ct.to_csv(f"{a.out}/T2_contrasts.csv", index=False)

A.variance_components(d).to_csv(f"{a.out}/T4_variance_components.csv", index=False)
if a.permutations:
    json.dump(A.permutation_ranking_test(d, B=a.permutations),
              open(f"{a.out}/ranking_test.json", "w"), indent=1)
print(ct.round(3).to_string(index=False))
