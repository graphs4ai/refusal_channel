"""01 — downloads the sources, builds the pool and selects probes + calibration."""
import json, argparse
import pandas as pd
from sycodef import items as I

ap = argparse.ArgumentParser()
ap.add_argument("--n-probe", type=int, default=300)
ap.add_argument("--n-cal", type=int, default=3)
ap.add_argument("--max-cos", type=float, default=0.15)
ap.add_argument("--out", default="out")
a = ap.parse_args()

mp = I.download(I.MMLU_PRO_URL, "data_raw/mmlu_pro_test.parquet")
tq = I.download(I.TQA_GEN_URL, "data_raw/tqa_gen.parquet")
pool = pd.concat([I.build_mmlu_items(mp), I.build_tqa_items(tq)], ignore_index=True)
probes, cal, cmap, report = I.assign_calibration(pool, a.n_probe, a.n_cal, a.max_cos)
probes.to_csv(f"{a.out}/item_pool_probes.csv", index=False)
cal.to_csv(f"{a.out}/item_pool_calibration.csv", index=False)
json.dump(cmap, open(f"{a.out}/calibration_map.json", "w"), ensure_ascii=False, indent=1)
json.dump({"n_pool": int(len(pool)), **report},
          open(f"{a.out}/item_pool_report.json", "w"), indent=1)
print(json.dumps({"n_pool": int(len(pool)), **report}, indent=1))
