"""03c — runs a SINGLE shard, with a wall-clock time cap."""
import argparse, json, time
import pandas as pd
from sycodef import collect as C, runner as R, shards as S

ap = argparse.ArgumentParser()
ap.add_argument("--shard-id", required=True)
ap.add_argument("--max-hours", type=float, default=5.0)
ap.add_argument("--backend", choices=["vllm", "hf", "stub", "openai", "anthropic"],
                default="vllm")
ap.add_argument("--batch-size", type=int, default=48)
ap.add_argument("--config", default="config/models_3090.json")
ap.add_argument("--base-url", default="https://api.openai.com/v1")
ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--out", default="out")
a = ap.parse_args()

cfg = json.load(open(a.config))
man = pd.read_csv(f"{a.out}/shard_manifest.csv").set_index("shard_id")
row = man.loc[a.shard_id]
design = pd.read_csv(f"{a.out}/design.csv")
sd = S.shard_design(design, {**row.to_dict(), "shard_id": a.shard_id})
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")
mc = cfg["models"].get(row["model"], {})
ledger = S.ShardLedger(f"{a.out}/shard_ledger.jsonl")
params = {"temperature": 0.0, "max_tokens": cfg["defaults"]["max_tokens"]}
cache = R.Cache(f"{a.out}/responses_{mc.get('alias', 'model')}.sqlite")

ledger.append(shard_id=a.shard_id, status="running", n_records=int(len(sd)))
t0 = time.time()
try:
    if a.backend in ("vllm", "hf", "stub"):
        if a.backend == "vllm":
            from sycodef.backends_gpu import VLLMBackend as B
            backend = B(row["model"], dtype=mc.get("dtype", "bfloat16"),
                        quantization=mc.get("quantization"),
                        max_model_len=cfg["defaults"]["max_model_len"],
                        gpu_memory_utilization=mc.get("gpu_memory_utilization", 0.9),
                        max_num_seqs=mc.get("max_num_seqs", 48),
                        enable_prefix_caching=cfg["defaults"]["enable_prefix_caching"],
                        chat_template_kwargs=mc.get("chat_template_kwargs"))
        elif a.backend == "hf":
            from sycodef.backends_gpu import HFBackend as B
            backend = B(row["model"], dtype=mc.get("dtype", "bfloat16"))
        else:
            from sycodef.backends_gpu import BatchedStub as B
            backend = B()
        summary = C.collect_batched(sd, probes, cal, backend, cache, params=params,
                                    batch_size=a.batch_size,
                                    max_seconds=a.max_hours * 3600)
    else:
        backend = (R.OpenAICompatBackend(a.base_url, a.api_key_env)
                   if a.backend == "openai" else R.AnthropicBackend(a.api_key_env))
        summary = C.collect(sd, probes, cal, backend, cache, params=params,
                            max_workers=a.workers)
        summary["stopped_early"] = False
    elapsed = time.time() - t0
    n_new = summary["pass1"]["new"] + summary["pass2"]["new"]
    n_err = summary["pass1"]["errors"] + summary["pass2"]["errors"]
    ledger.append(shard_id=a.shard_id,
                  status="partial" if summary["stopped_early"] else "done",
                  elapsed_s=round(elapsed, 1), n_new=n_new, n_errors=n_err,
                  summary=summary)
    print(json.dumps({"shard": a.shard_id, "elapsed_h": round(elapsed / 3600, 2),
                      **summary}, indent=1, default=str))
except Exception as e:
    ledger.append(shard_id=a.shard_id, status="failed",
                  elapsed_s=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
    raise
