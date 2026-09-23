"""Bridge between the design table and the runner: assembles the messages for
each record and orders collection into two passes.

Pass 1 — everything that does not depend on a model response (baseline with no
pressure, neutral rephrasing, all `assert` cells). Pass 2 — the `doubt`
condition, which needs the model's own baseline answer as an assistant turn.
Pass 2 reads the baseline from the cache, so it never duplicates generation.
"""
from __future__ import annotations

import json

import pandas as pd

from . import design as D
from . import prompts as P
from .extract import parse_answer
from .runner import (Cache, cache_key, merged_params, run_records,
                     run_records_batched)


def _item_dicts(probes: pd.DataFrame, cal_pool: pd.DataFrame):
    return ({r["item_id"]: r.to_dict() for _, r in probes.iterrows()},
            {r["item_id"]: r.to_dict() for _, r in cal_pool.iterrows()})


def build_jobs(design: pd.DataFrame, probes: pd.DataFrame, cal_pool: pd.DataFrame,
               baseline_letters: dict | None = None, recognition_probe=False,
               implicit_history=False):
    """Generates the list of jobs for the runner. `baseline_letters`:
    (model, lang, item_id, option_order) -> letter, needed only for `doubt`.
    """
    pdict, cdict = _item_dicts(probes, cal_pool)
    # A round-trip through the CSV converts the empty string of
    # `asserted_letter` into NaN, which is *truthy* and leaks into the prompt
    # and into the backend meta. Normalize here, at the entry point, rather
    # than in every consumer.
    design = design.copy()
    if "asserted_letter" in design:
        design["asserted_letter"] = design["asserted_letter"].fillna("").astype(str)
    jobs, skipped = [], 0
    for _, r in design.iterrows():
        item = D.apply_option_order(pdict[r["item_id"]], r["option_order"])
        cal_items = []
        if r["history"] != "none":
            ids = json.loads(r["calibration_ids"])
            base = [cdict[i] for i in ids]
            if r["history"] == "filler":
                cal_items = base          # presented without user assertion
            else:
                n_correct = int(str(r["history"]).split("/")[0])
                cal_items = D.calibration_claims(base, n_correct, r["cal_rotation"])
        bl = None
        if r["pressure"] == "doubt":
            if not baseline_letters:
                skipped += 1
                continue
            bl = baseline_letters.get((r["model"], r["lang"], r["item_id"],
                                       r["option_order"]))
            if bl is None:
                skipped += 1
                continue
        msgs = P.build_messages(r, item, cal_items, lang=r["lang"],
                                para=r["paraphrase"], baseline_answer=bl,
                                recognition_probe=recognition_probe,
                                implicit_history=implicit_history)
        jparams = {}
        if "temperature" in r and pd.notna(r["temperature"]):
            jparams["temperature"] = float(r["temperature"])
        if "seed" in r and pd.notna(r.get("seed")):
            jparams["seed"] = int(r["seed"])
        jobs.append({"record_id": r["record_id"], "model": r["model"],
                     "messages": msgs, "params": jparams,
                     # orders batch collection: records that share the
                     # calibration prefix end up adjacent so vLLM's prefix
                     # cache pays for that prefill only once
                     "prefix_key": f'{r["lang"]}|{r["item_id"]}|{r["history"]}|'
                                   f'{r["option_order"]}|{r["paraphrase"]}',
                     "meta": {"true_letter": "ABCD"[int(item["answer_index"])],
                              "asserted_letter": r["asserted_letter"],
                              "history": r["history"], "signal": r["signal"],
                              "position": r["position"]}})
    return jobs, skipped


def baseline_letters_from_cache(design: pd.DataFrame, probes, cal_pool,
                                cache: Cache, params: dict) -> dict:
    """Mode of the baseline answers by (model, lang, item, option_order)."""
    base = design[(design["pressure"] == "none") & (design["history"] == "none")]
    jobs, _ = build_jobs(base, probes, cal_pool)
    key_of = {j["record_id"]: cache_key(j["model"], j["messages"],
                                       merged_params(j, params)) for j in jobs}
    rows = []
    for _, r in base.iterrows():
        got = cache.get(key_of.get(r["record_id"], ""))
        if not got or not got[0]:
            continue
        letter, _rule = parse_answer(got[0])
        if letter:
            rows.append({"model": r["model"], "lang": r["lang"],
                         "item_id": r["item_id"],
                         "option_order": r["option_order"], "answer": letter})
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    mode = (df.groupby(["model", "lang", "item_id", "option_order"])["answer"]
            .agg(lambda s: s.value_counts().idxmax()))
    return mode.to_dict()


def collect(design: pd.DataFrame, probes, cal_pool, backend, cache: Cache,
            params: dict | None = None, max_workers=8, **kw):
    """Collects in two passes. Returns the summary of each pass."""
    params = params or {"temperature": 0.0, "max_tokens": 64}
    pass1 = design[design["pressure"] != "doubt"]
    j1, _ = build_jobs(pass1, probes, cal_pool, **kw)
    r1 = run_records(j1, backend, cache, max_workers=max_workers, params=params)

    bl = baseline_letters_from_cache(design, probes, cal_pool, cache, params)
    pass2 = design[design["pressure"] == "doubt"]
    j2, skipped = build_jobs(pass2, probes, cal_pool, baseline_letters=bl, **kw)
    r2 = run_records(j2, backend, cache, max_workers=max_workers, params=params)
    return {"pass1": dict(zip(["cached", "new", "errors"], r1)), "n_jobs1": len(j1),
            "pass2": dict(zip(["cached", "new", "errors"], r2)), "n_jobs2": len(j2),
            "doubt_skipped_no_baseline": skipped}


def responses_frame(design: pd.DataFrame, probes, cal_pool, cache: Cache,
                    params: dict | None = None, **kw) -> pd.DataFrame:
    """Joins the design with the cached texts and parses them."""
    from .extract import parse_frame
    params = params or {"temperature": 0.0, "max_tokens": 64}
    bl = baseline_letters_from_cache(design, probes, cal_pool, cache, params)
    jobs, _ = build_jobs(design, probes, cal_pool, baseline_letters=bl, **kw)
    recs = []
    for j in jobs:
        got = cache.get(cache_key(j["model"], j["messages"], merged_params(j, params)))
        recs.append({"record_id": j["record_id"],
                     "text": got[0] if got else None,
                     "error": got[1] if got else "missing",
                     "n_messages": len(j["messages"]),
                     "prompt_chars": sum(len(m["content"]) for m in j["messages"])})
    out = design.merge(pd.DataFrame(recs), on="record_id", how="inner")
    return parse_frame(out)


def collect_batched(design: pd.DataFrame, probes, cal_pool, backend, cache: Cache,
                    params: dict | None = None, batch_size: int = 64,
                    max_seconds: float | None = None, **kw):
    """Collects in two passes for local batch backends (vLLM/HF).

    The time budget is split between the passes in proportion to the number
    of jobs, so that pass 2 (`doubt`, which depends on the baseline from
    pass 1) is not always left without time. Returns the same summary as
    `collect`, plus `stopped_early`.
    """
    params = params or {"temperature": 0.0, "max_tokens": 64}
    pass1 = design[design["pressure"] != "doubt"]
    pass2_design = design[design["pressure"] == "doubt"]
    j1, _ = build_jobs(pass1, probes, cal_pool, **kw)
    share = len(j1) / max(len(j1) + len(pass2_design), 1)
    b1 = max_seconds * share if max_seconds else None
    r1 = run_records_batched(j1, backend, cache, params=params,
                             batch_size=batch_size, max_seconds=b1)

    bl = baseline_letters_from_cache(design, probes, cal_pool, cache, params)
    j2, skipped = build_jobs(pass2_design, probes, cal_pool,
                             baseline_letters=bl, **kw)
    b2 = (max_seconds - b1) if max_seconds else None
    r2 = run_records_batched(j2, backend, cache, params=params,
                             batch_size=batch_size, max_seconds=b2)
    return {"pass1": dict(zip(["cached", "new", "errors", "stopped"], r1)),
            "n_jobs1": len(j1),
            "pass2": dict(zip(["cached", "new", "errors", "stopped"], r2)),
            "n_jobs2": len(j2),
            "doubt_skipped_no_baseline": skipped,
            "stopped_early": bool(r1[3] or r2[3])}
