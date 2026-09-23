"""03b — shard plan with per-run hour cap."""
import argparse, json, os
import pandas as pd
from sycodef import shards as S

ap = argparse.ArgumentParser()
ap.add_argument("--target-hours", type=float, default=4.0)
ap.add_argument("--max-hours", type=float, default=6.0)
ap.add_argument("--config", default="config/models_3090.json")
ap.add_argument("--sample-items", type=int, default=12)
ap.add_argument("--max-records-per-shard", type=int, default=5000)
ap.add_argument("--out", default="out")
a = ap.parse_args()

cfg = json.load(open(a.config))
design = pd.read_csv(f"{a.out}/design.csv")
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")

# tokens don't depend on the model under the chars/4 heuristic: estimate only one
one_model = design[design["model"] == design["model"].iloc[0]]
tokens = S.estimate_tokens_by_cell(one_model, probes, cal, sample_items=a.sample_items)
tokens.to_csv(f"{a.out}/tokens_by_cell.csv", index=False)

measured = {}
mp = f"{a.out}/throughput.json"
if os.path.exists(mp):
    measured = json.load(open(mp))

tps = {}
for m in design["model"].unique():
    if m in measured:
        r = measured[m]
        tps[m] = S.Throughput(model=m, prefill_tok_s=r["prefill_tok_s"],
                              decode_tok_s=r["decode_tok_s"], source="measured",
                              note=f"{r['records_per_s']} records/s")
    else:
        mc = cfg["models"].get(m)
        if mc is None:
            raise SystemExit(f"{m} is not in {a.config} nor in throughput.json")
        tps[m] = S.roofline_throughput(m, mc["params_b"], mc["quant_bits"])

man = S.plan_shards(design, tokens, tps, target_hours=a.target_hours,
                    max_hours=a.max_hours,
                    max_records_per_shard=a.max_records_per_shard)
man.to_csv(f"{a.out}/shard_manifest.csv", index=False)
cols = ["shard_id", "n_items", "n_records", "est_hours", "throughput_source"]
print(man[cols].to_string(index=False))
print(f"\ntotal: {len(man)} shards, {man.est_hours.sum():.1f} h estimated, "
      f"{man.n_records.sum()} generations")
