"""03d — queue: runs pending shards of ONE model until the budget runs out.

Grouping by model matters: vLLM loads the weights once per process, and
reloading for every shard costs 1–3 min. The budget is wall-clock; when it
runs out, the current shard finishes the ongoing batch and the process exits
with the ledger consistent.
"""
import argparse, subprocess, sys, time
import pandas as pd
from sycodef import shards as S

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budget-hours", type=float, default=5.0)
ap.add_argument("--backend", default="vllm")
ap.add_argument("--batch-size", type=int, default=48)
ap.add_argument("--out", default="out")
a = ap.parse_args()

man = pd.read_csv(f"{a.out}/shard_manifest.csv")
ledger = S.ShardLedger(f"{a.out}/shard_ledger.jsonl")
pending = ledger.pending(man, model=a.model)
print(f"{len(pending)} pending shards for {a.model}")

t0 = time.time()
# minimum headroom for it to be worth reloading the weights again; never
# greater than the budget itself, otherwise the queue exits without running anything
headroom = min(0.25, a.budget_hours * 0.2)
for _, r in pending.iterrows():
    left = a.budget_hours - (time.time() - t0) / 3600
    if left <= headroom:
        print(f"budget exhausted; {r.shard_id} is left for the next run")
        break
    cmd = [sys.executable, "scripts/03c_run_shard.py", "--shard-id", r.shard_id,
           "--max-hours", f"{min(left, r.est_hours * 1.5):.2f}",
           "--backend", a.backend, "--batch-size", str(a.batch_size),
           "--out", a.out]
    print(" ".join(cmd), flush=True)
    if subprocess.call(cmd) != 0:
        print(f"shard {r.shard_id} failed; check the ledger", file=sys.stderr)
        break
print(S.ShardLedger(f"{a.out}/shard_ledger.jsonl").report(man).to_string(index=False))
