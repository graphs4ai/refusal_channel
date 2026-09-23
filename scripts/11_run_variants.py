"""11 — prompt robustness arm for the refusal outcome.

Reduced grid (pressure arm only, 1 replicate) run under the four
variants of `sycodef.variants`. The question is whether the refusal
survives when the pre-filled ack changes shape, when it disappears, and
when a letter is explicitly required.
"""
import argparse, json, time
import pandas as pd
from sycodef import collect as C, runner as R, variants as V, extract as E

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--config", default="config/models_3090.json")
ap.add_argument("--n-items", type=int, default=100)
ap.add_argument("--batch-size", type=int, default=48)
ap.add_argument("--backend", choices=["vllm", "stub"], default="vllm")
ap.add_argument("--out", default="out")
ap.add_argument("--variants", default=",".join(V.VARIANTS))
a = ap.parse_args()

cfg = json.load(open(a.config))
mc = cfg["models"].get(a.model, {})
design = pd.read_csv(f"{a.out}/design.csv")
d = design[(design.model == a.model) & (design.pressure == "assert")
           & (design.rep == 0)].copy()
items = sorted(d.item_id.unique())[:a.n_items]
d = d[d.item_id.isin(items)]
if d.empty:
    raise SystemExit(f"no assert records for {a.model} in {a.out}/design.csv")
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")

base_jobs, _ = C.build_jobs(d, probes, cal)
by_rid = {r.record_id: r for r in d.itertuples()}

def make_backend():
    """Only built if some generation is missing: when the cache already
    covers everything (rerun after an analysis failure), loading 16 GB of
    weights is pure waste."""
    if a.backend == "stub":
        from sycodef.backends_gpu import BatchedStub
        return BatchedStub()
    from sycodef.backends_gpu import VLLMBackend
    return VLLMBackend(a.model, dtype=mc.get("dtype", "bfloat16"),
                       quantization=mc.get("quantization"),
                       max_model_len=cfg["defaults"]["max_model_len"],
                       gpu_memory_utilization=mc.get("gpu_memory_utilization", 0.9),
                       max_num_seqs=mc.get("max_num_seqs", 48),
                       chat_template_kwargs=mc.get("chat_template_kwargs"))

alias = mc.get("alias", "model")
cache = R.Cache(f"{a.out}/variants_{alias}.sqlite")
jobs = []
for v in a.variants.split(","):
    for j in base_jobs:
        r = by_rid[j["record_id"]]
        # the variant goes into params because it goes into the cache key: under
        # history='none' the v0/v1/v2 variants produce identical messages
        # (there is no assistant turn to alter) and would collide into a single row
        jobs.append({**j,
                     "params": {**j.get("params", {}), "variant": v},
                     "record_id": f'{j["record_id"]}__{v}',
                     "messages": V.apply_variant(j["messages"], v,
                                                 para=r.paraphrase, lang=r.lang),
                     "prefix_key": f'{v}|{j["prefix_key"]}'})

t0 = time.time()
missing = [j for j in jobs
           if cache.get(R.cache_key(j["model"], j["messages"],
                                    R.merged_params(j, {"temperature": 0.0,
                                                        "max_tokens": cfg["defaults"]["max_tokens"]}))) is None]
backend = make_backend() if missing else None
hits, new, err, _ = (len(jobs), 0, 0, False) if not missing else R.run_records_batched(
    jobs, backend, cache,
    params={"temperature": 0.0, "max_tokens": cfg["defaults"]["max_tokens"]},
    batch_size=a.batch_size, progress_every=0)

df = cache.to_dataframe()
# Export only the rows that correspond to the CURRENT prompt construction.
# The cache also holds rows from superseded versions of a variant (e.g. the
# first v2_no_ack build, which did not merge the probe into the preceding user
# turn); they share the record_id but not the cache key, and counting both
# would mix two prompt structures in one cell.
_p = {"temperature": 0.0, "max_tokens": cfg["defaults"]["max_tokens"]}
current_keys = {R.cache_key(j["model"], j["messages"], R.merged_params(j, _p))
                for j in jobs}
n_all = len(df)
df = df[df["key"].isin(current_keys) & df["error"].isna()].copy()
print(f"exported {len(df)} current-version rows (dropped {n_all - len(df)} "
      f"superseded or failed rows)")
# an empty generation becomes NaN on the round-trip through SQLite and breaks the parser
df["text"] = df["text"].fillna("").astype(str)
df["variant"] = df.record_id.str.rsplit("__", n=1).str[-1]
df["base_record_id"] = df.record_id.str.rsplit("__", n=1).str[0]
keys = ["record_id", "item_id", "history", "signal", "position", "paraphrase",
        "option_order", "asserted_letter", "model", "lang"]
df = df.merge(d[keys].rename(columns={"record_id": "base_record_id"}),
              on="base_record_id", how="inner")
df = E.parse_frame(df, text_col="text")
df["refused"] = df.answer.isna().astype(int)
df.to_csv(f"{a.out}/variants_{alias}.csv", index=False)
print(json.dumps({"model": a.model, "n_jobs": len(jobs), "cached": hits,
                  "new": new, "errors": err,
                  "elapsed_min": round((time.time() - t0) / 60, 1)}, indent=1))
print(df.groupby(["variant", "history"])["refused"].mean().round(3).unstack().to_string())
