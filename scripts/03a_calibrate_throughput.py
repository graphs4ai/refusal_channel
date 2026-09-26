"""03a — measures the actual backend throughput on a small sample.

Runs ~N records from the design and writes `out/throughput.json`. The shard
plan (03b) must be redone after this: the roofline estimates are only for
sizing before you have the machine.
"""
import argparse, json, time
import pandas as pd
from sycodef import collect as C, runner as R, shards as S

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="HF id or API model name")
ap.add_argument("--config", default="config/models_3090.json")
ap.add_argument("--n-records", type=int, default=200)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--backend", choices=["vllm", "hf", "stub"], default="vllm")
ap.add_argument("--out", default="out")
a = ap.parse_args()

cfg = json.load(open(a.config))
mc = cfg["models"].get(a.model, {})
design = pd.read_csv(f"{a.out}/design.csv")
design = design[design["model"] == a.model]
if design.empty:
    raise SystemExit(f"no records for {a.model} in {a.out}/design.csv")
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")

# stratified sample by cell: short and long prompts both enter the measurement
sample = (design.groupby("cell_id", group_keys=False)
          .apply(lambda g: g.head(max(a.n_records // design["cell_id"].nunique(), 1))))
if a.backend == "vllm":
    from sycodef.backends_gpu import VLLMBackend
    backend = VLLMBackend(a.model, dtype=mc.get("dtype", "bfloat16"),
                          quantization=mc.get("quantization"),
                          max_model_len=cfg["defaults"]["max_model_len"],
                          gpu_memory_utilization=mc.get("gpu_memory_utilization", 0.90),
                          max_num_seqs=mc.get("max_num_seqs", 48),
                          chat_template_kwargs=mc.get("chat_template_kwargs"))
elif a.backend == "hf":
    from sycodef.backends_gpu import HFBackend
    backend = HFBackend(a.model, dtype=mc.get("dtype", "bfloat16"))
else:
    from sycodef.backends_gpu import BatchedStub
    backend = BatchedStub()

jobs, _ = C.build_jobs(sample, probes, cal,
                       baseline_letters={(r.model, r.lang, r.item_id, r.option_order): "A"
                                         for r in sample.itertuples()})
tok = getattr(backend, "tok", None)
def _n_prefill(msgs):
    # The Gemma-2 template rejects role=system ("System role not supported"):
    # the prefill count needs the same normalization the backend applies,
    # otherwise the shard dies here instead of during generation.
    from sycodef.backends_gpu import normalize_messages, supports_system_role
    msgs = normalize_messages(msgs, supports_system_role(a.model))
    # transformers>=4.46: tokenize=True returns a BatchEncoding (len == number of keys),
    # not the list of ids — without return_dict=False the measured prefill becomes 2.
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                  return_dict=False)
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return len(enc)

prefill = [_n_prefill(j["messages"]) for j in jobs] if tok is not None \
    else [sum(len(m["content"]) for m in j["messages"]) / 4 for j in jobs]

cache = R.Cache(f"{a.out}/calib_{mc.get('alias', 'model')}.sqlite")
t0 = time.time()
hits, new, err, _ = R.run_records_batched(
    jobs, backend, cache, params={"temperature": 0.0,
                                  "max_tokens": cfg["defaults"]["max_tokens"]},
    batch_size=a.batch_size, progress_every=0)
elapsed = time.time() - t0

out_tok = cfg["defaults"]["max_tokens"] * 0.4          # actual output is short
rec_s = new / max(elapsed, 1e-9)
res = {"model": a.model, "backend": a.backend, "n_records": int(new),
       "elapsed_s": round(elapsed, 1), "records_per_s": round(rec_s, 3),
       "mean_prefill_tokens": round(float(sum(prefill) / len(prefill)), 1),
       "prefill_tok_s": round(rec_s * float(sum(prefill) / len(prefill)), 1),
       "decode_tok_s": round(rec_s * out_tok, 1), "errors": int(err),
       "source": "measured"}
path = f"{a.out}/throughput.json"
allt = json.load(open(path)) if __import__("os").path.exists(path) else {}
allt[a.model] = res
json.dump(allt, open(path, "w"), indent=1)
print(json.dumps(res, indent=1))
