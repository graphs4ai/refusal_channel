"""02 — registry grid and generation budget."""
import json, argparse
import pandas as pd
from sycodef import design as D

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--n-shared", type=int, default=300)
ap.add_argument("--n-p1", type=int, default=100)
ap.add_argument("--reps-shared", type=int, default=6)
ap.add_argument("--reps-pressure", type=int, default=5)
ap.add_argument("--full-history", action="store_true")
ap.add_argument("--langs", nargs="+", default=["en"])
ap.add_argument("--out", default="out")
a = ap.parse_args()

probes = pd.read_csv(f"{a.out}/item_pool_probes.csv").head(a.n_shared)
cmap = json.load(open(f"{a.out}/calibration_map.json"))
hist = D.HISTORY_FULL if a.full_history else D.HISTORY_MIN
d = D.build_design(probes, cmap, a.models, history_levels=hist, langs=tuple(a.langs),
                   reps_shared=a.reps_shared, reps_pressure=a.reps_pressure,
                   p1_items=list(probes["item_id"][:a.n_p1]))
d.to_csv(f"{a.out}/design.csv", index=False)
b = D.budget(d)
b.to_csv(f"{a.out}/budget.csv", index=False)
print(b.to_string(index=False))
