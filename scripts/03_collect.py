"""03 — two-pass collection, with resumable cache."""
import argparse
import pandas as pd
from sycodef import collect as C, runner as R

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["stub", "openai", "anthropic"], default="stub")
ap.add_argument("--base-url", default="https://api.openai.com/v1")
ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--max-tokens", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit-items", type=int, default=0, help="smoke test: use N items")
ap.add_argument("--cache", default="out/responses.sqlite")
ap.add_argument("--out", default="out")
a = ap.parse_args()

design = pd.read_csv(f"{a.out}/design.csv")
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")
if a.limit_items:
    keep = set(probes["item_id"][:a.limit_items])
    design = design[design["item_id"].isin(keep)]

backend = {"stub": lambda: R.StubBackend(),
           "openai": lambda: R.OpenAICompatBackend(a.base_url, a.api_key_env),
           "anthropic": lambda: R.AnthropicBackend(a.api_key_env)}[a.backend]()
params = {"temperature": a.temperature, "max_tokens": a.max_tokens}
cache = R.Cache(a.cache)
print(C.collect(design, probes, cal, backend, cache, params=params, max_workers=a.workers))
df = C.responses_frame(design, probes, cal, cache, params=params)
df.to_csv(f"{a.out}/responses.csv", index=False)
print("records:", len(df), "| parse ok:", round(float(df["answer"].notna().mean()), 4))
