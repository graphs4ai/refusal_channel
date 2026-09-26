"""06 — progress report and consolidation of cached responses."""
import argparse, glob, json
import pandas as pd
from sycodef import collect as C, runner as R, shards as S

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--merge", action="store_true", help="also writes responses.csv")
a = ap.parse_args()

man = pd.read_csv(f"{a.out}/shard_manifest.csv")
rep = S.ShardLedger(f"{a.out}/shard_ledger.jsonl").report(man)
rep.to_csv(f"{a.out}/shard_progress.csv", index=False)
print(rep.to_string(index=False))
done = (rep.status == "done").sum()
print(f"\n{done}/{len(rep)} shards completed | "
      f"{rep.elapsed_h.sum():.1f} h spent of {man.est_hours.sum():.1f} h estimated")

if a.merge:
    design = pd.read_csv(f"{a.out}/design.csv")
    probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
    cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")
    frames = []
    for db in sorted(glob.glob(f"{a.out}/responses_*.sqlite")):
        cache = R.Cache(db)
        models = pd.read_sql_query("SELECT DISTINCT model FROM responses",
                                   cache._conn)["model"].tolist()
        sub = design[design["model"].isin(models)]
        if len(sub):
            frames.append(C.responses_frame(sub, probes, cal, cache))
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(f"{a.out}/responses.csv", index=False)
        print(f"responses.csv: {len(df)} records | "
              f"parse ok {df.answer.notna().mean():.4f}")
