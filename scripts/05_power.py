"""05 — recovery of estimators and power curves."""
import argparse
from sycodef import simulate as S

ap = argparse.ArgumentParser()
ap.add_argument("--sims", type=int, default=150)
ap.add_argument("--out", default="out")
a = ap.parse_args()

rec = S.recovery_check(n_sims=max(a.sims // 3, 10))
rec.to_csv(f"{a.out}/validacao_recuperacao.csv", index=False)
print(rec.round(3).to_string(index=False))

import pandas as pd
curves = [S.power_curve(n_sims=a.sims, n_items=n, reps=r)
          for n, r in [(100, 3), (150, 3), (100, 5)]]
pw = pd.concat(curves, ignore_index=True)
pw.to_csv(f"{a.out}/power_h2.csv", index=False)
print(pw.round(3).to_string(index=False))
