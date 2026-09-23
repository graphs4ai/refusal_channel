"""Partitioning the collection into runs with a wall-clock time cap.

Operating rule: **no run exceeds `target_hours`** (default 4 h, cap 6 h). A
shard is the triple (model, arm, item block) — self-contained by
construction:

* the shared arm carries the three cells of the same item block, and the
  `doubt` condition needs the baseline of the **same** items, which is in the
  same shard; the two-pass collection in `collect.collect` closes within the
  shard;
* the P1 arm has no generation dependency between cells (the baseline is only
  used in the analysis), so any item block is independent.

The ledger (`out/shard_ledger.jsonl`, append-only) records status, time and
counts per shard. Since the SQLite cache is keyed by content, re-running an
interrupted shard only costs what was missing.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

# RTX 3090: 35.6 TFLOP/s bf16 with FP32 accumulation, 936 GB/s bandwidth.
RTX3090 = {"tflops_bf16": 35.6, "mem_bw_gb_s": 936.0}


@dataclass
class Throughput:
    """Throughput of a model on a given target. `source` tells whether it was
    measured or estimated — the shard plan should be redone after calibration."""
    model: str
    prefill_tok_s: float
    decode_tok_s: float
    source: str = "roofline"          # "measured" | "roofline" | "api"
    load_s: float = 120.0
    note: str = ""

    def hours(self, prefill_tokens: float, output_tokens: float) -> float:
        return (self.load_s + prefill_tokens / self.prefill_tok_s
                + output_tokens / self.decode_tok_s) / 3600.0

    def records_for_hours(self, hours: float, prefill_per_rec: float,
                          out_per_rec: float) -> int:
        per_rec = prefill_per_rec / self.prefill_tok_s + out_per_rec / self.decode_tok_s
        return max(int((hours * 3600.0 - self.load_s) / per_rec), 1)


def roofline_throughput(model: str, params_b: float, quant_bits: int = 16,
                        mfu: float = 0.35, batch: int = 64,
                        hw: dict = RTX3090) -> Throughput:
    """Ceiling estimate for planning before having the machine.

    Prefill is FLOP-bound (2*N per token); batched decode is memory-bandwidth
    bound (one pass over the weights per step, amortized over the batch).
    Deliberately conservative: it serves to size shards, not to promise
    times. `scripts/03a_calibrate_throughput.py` replaces it with a measurement.
    """
    bytes_per_param = quant_bits / 8.0
    prefill = hw["tflops_bf16"] * 1e12 * mfu / (2 * params_b * 1e9)
    decode_mem = batch * hw["mem_bw_gb_s"] * 1e9 / (bytes_per_param * params_b * 1e9)
    decode = min(decode_mem, prefill)
    return Throughput(model=model, prefill_tok_s=prefill, decode_tok_s=decode,
                      source="roofline",
                      note=f"{params_b}B @ {quant_bits}-bit, mfu={mfu}, batch={batch}")


def api_throughput(model: str, records_per_s: float, prefill_per_rec: float,
                   out_per_rec: float) -> Throughput:
    """For API backends, the bottleneck is concurrency and rate limit, not
    FLOPs. Converts an observed records/s rate into the same interface."""
    return Throughput(model=model,
                      prefill_tok_s=records_per_s * prefill_per_rec * 2,
                      decode_tok_s=records_per_s * out_per_rec * 2,
                      source="api", load_s=0.0,
                      note=f"{records_per_s:.2f} records/s observed")


# ---------------------------------------------------------------- tokens
def estimate_tokens_by_cell(design: pd.DataFrame, probes: pd.DataFrame,
                            cal_pool: pd.DataFrame, sample_items: int = 12,
                            chars_per_token: float = 4.0,
                            out_tokens: float = 25.0,
                            tokenizer=None) -> pd.DataFrame:
    """Input tokens per cell, measured on a sample of items and applied to
    the whole design. With `tokenizer` (HF), counts real tokens; without it,
    uses chars/4, which slightly overestimates in English."""
    from .collect import build_jobs
    items = list(dict.fromkeys(design["item_id"]))[:sample_items]
    sample = design[design["item_id"].isin(items)]
    jobs, _ = build_jobs(sample, probes, cal_pool,
                         baseline_letters={(r.model, r.lang, r.item_id,
                                            r.option_order): "A"
                                           for r in sample.itertuples()})
    by_rec = {j["record_id"]: j for j in jobs}
    rows = []
    for _, r in sample.iterrows():
        j = by_rec.get(r["record_id"])
        if j is None:
            continue
        if tokenizer is not None:
            n = len(tokenizer.apply_chat_template(j["messages"], tokenize=True,
                                                  add_generation_prompt=True))
        else:
            n = sum(len(m["content"]) for m in j["messages"]) / chars_per_token
        rows.append({"cell_id": r["cell_id"], "lang": r["lang"],
                     "prefill_tokens": float(n)})
    t = (pd.DataFrame(rows).groupby(["cell_id", "lang"], as_index=False)
         ["prefill_tokens"].mean())
    t["output_tokens"] = out_tokens
    t.attrs["source"] = "tokenizer" if tokenizer is not None else "chars/4"
    return t


def attach_token_costs(design: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
    d = design.merge(tokens, on=["cell_id", "lang"], how="left")
    miss = int(d["prefill_tokens"].isna().sum())
    if miss:
        d["prefill_tokens"] = d["prefill_tokens"].fillna(tokens["prefill_tokens"].mean())
        d["output_tokens"] = d["output_tokens"].fillna(tokens["output_tokens"].mean())
    return d


# ---------------------------------------------------------------- shards
def plan_shards(design: pd.DataFrame, tokens: pd.DataFrame,
                throughputs: dict, target_hours: float = 4.0,
                max_hours: float = 6.0,
                max_records_per_shard: int | None = 5000) -> pd.DataFrame:
    """Shard manifest. A shard = (model, arm, language, item block),
    sized to fit within `target_hours`.

    The cap on records (`max_records_per_shard`) exists because the
    throughput estimate can be off by a factor of 2-3 before calibration; it
    keeps the size of each run bounded even when the estimated time is
    wrong. The plan fails loudly if any shard exceeds `max_hours`."""
    d = attach_token_costs(design, tokens)
    rows = []
    for (model, arm, lang), g in d.groupby(["model", "arm", "lang"], sort=False):
        tp = throughputs[model]
        items = list(dict.fromkeys(g["item_id"]))
        per_item = g.groupby("item_id")[["prefill_tokens", "output_tokens"]].sum()
        total_h = tp.hours(per_item["prefill_tokens"].sum(),
                           per_item["output_tokens"].sum())
        n_blocks = max(math.ceil(total_h / target_hours), 1)
        if max_records_per_shard:
            n_blocks = max(n_blocks, math.ceil(len(g) / max_records_per_shard))
        n_blocks = min(n_blocks, len(items))
        blocks = np.array_split(np.array(items), n_blocks)
        for b, block in enumerate(blocks):
            sub = per_item.loc[list(block)]
            h = tp.hours(sub["prefill_tokens"].sum(), sub["output_tokens"].sum())
            n_rec = int(g[g["item_id"].isin(block)].shape[0])
            rows.append({
                "shard_id": f"{model}__{arm}__{lang}__b{b:02d}",
                "model": model, "arm": arm, "lang": lang, "block": b,
                "n_items": len(block), "n_records": n_rec,
                "prefill_tokens": float(sub["prefill_tokens"].sum()),
                "output_tokens": float(sub["output_tokens"].sum()),
                "est_hours": round(float(h), 2),
                "throughput_source": tp.source,
                "items": json.dumps(list(block)),
            })
    man = pd.DataFrame(rows)
    over = man[man["est_hours"] > max_hours]
    if len(over):
        raise ValueError(
            f"{len(over)} shards above {max_hours} h even after the cut: "
            f"{over['shard_id'].tolist()[:3]} — reduce target_hours, the number of "
            "items per block, or use a quantized model")
    return man.sort_values(["model", "arm", "block"]).reset_index(drop=True)


def shard_design(design: pd.DataFrame, row) -> pd.DataFrame:
    items = set(json.loads(row["items"]))
    return design[(design["model"] == row["model"]) & (design["arm"] == row["arm"])
                  & (design["lang"] == row["lang"])
                  & (design["item_id"].isin(items))].copy()


# ---------------------------------------------------------------- ledger
class ShardLedger:
    """Append-only in JSON Lines: survives process crashes and is readable
    with `tail`. The current state of a shard is the last line with its id."""

    def __init__(self, path: str = "out/shard_ledger.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, **rec):
        rec.setdefault("ts", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def state(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        st = {}
        for line in open(self.path):
            line = line.strip()
            if line:
                r = json.loads(line)
                st[r["shard_id"]] = r
        return st

    def pending(self, manifest: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
        st = self.state()
        done = {k for k, v in st.items() if v.get("status") == "done"}
        m = manifest[~manifest["shard_id"].isin(done)]
        return m[m["model"] == model] if model else m

    def report(self, manifest: pd.DataFrame) -> pd.DataFrame:
        st = self.state()
        m = manifest.copy()
        m["status"] = m["shard_id"].map(lambda s: st.get(s, {}).get("status", "pending"))
        m["elapsed_h"] = m["shard_id"].map(
            lambda s: round(st.get(s, {}).get("elapsed_s", 0) / 3600, 2))
        m["n_new"] = m["shard_id"].map(lambda s: st.get(s, {}).get("n_new", 0))
        m["n_errors"] = m["shard_id"].map(lambda s: st.get(s, {}).get("n_errors", 0))
        return m[["shard_id", "model", "arm", "n_records", "est_hours",
                  "status", "elapsed_h", "n_new", "n_errors"]]
