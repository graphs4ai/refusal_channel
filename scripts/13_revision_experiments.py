"""13 — additional generations requested during revision (one model per call).

E1  matched no-pressure control. For every (probe, history, replicate) of the
    pressure arm, the same calibration dialogue (none / filler / 3/3 / 0/3),
    option order and paraphrase are followed by a NEUTRAL probe with no user
    assertion. Because 3/3, 0/3 and filler histories do not depend on the
    calibration rotation, one control record is exactly matched to all four
    pressured cells (2 positions x 2 signals) of the same probe/history/rep.
    Output: out/rev_e1_<alias>.csv

E2  self-generated history ("v4_selfgen") on the robustness grid (first 100
    probes, replicate 0, the same 1,600 records per model as script 11). The
    pre-filled acknowledgements are replaced, turn by turn, by the evaluated
    model's own greedy reply to the preceding user / verification turn; the
    generated history is then fixed and the probe is answered once. User turns,
    verification turns, system prompt and probe are unchanged.
    Output: out/rev_e2_<alias>.csv (includes the generated assistant turns)

E3  (a) regenerates robustness-arm rows whose only cached attempt failed
        (v2_no_ack, history=none, Gemma-2 and Mistral);
    (b) re-runs every main-arm and robustness-arm record whose response was cut
        at the 64-token cap AND yielded no parsable option, with a 256-token
        cap, to separate truncation from non-answers.
    Output: out/rev_e3_trunc_<alias>.csv
"""
import argparse, json, time
import pandas as pd
from sycodef import collect as C, runner as R, variants as V, extract as E

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--config", default="config/models_3090.json")
ap.add_argument("--out", default="out")
ap.add_argument("--exp", default="e1,e2,e3")
ap.add_argument("--batch-size", type=int, default=512)
ap.add_argument("--n-items-robust", type=int, default=100)
ap.add_argument("--trunc-max-tokens", type=int, default=256)
ap.add_argument("--cache-dir", default=None,
                help="where caches/CSVs are read and written (default: --out)")
ap.add_argument("--backend", choices=["vllm", "stub"], default="vllm")
a = ap.parse_args()
CD = a.cache_dir or a.out
EXPS = set(a.exp.split(","))

cfg = json.load(open(a.config))
mc = cfg["models"][a.model]
alias = mc["alias"]
MAXT = cfg["defaults"]["max_tokens"]
PARAMS = {"temperature": 0.0, "max_tokens": MAXT}
design = pd.read_csv(f"{a.out}/design.csv", dtype={"asserted_letter": str})
design = design[design.model == a.model].copy()
design["asserted_letter"] = design["asserted_letter"].fillna("").astype(str)
probes = pd.read_csv(f"{a.out}/item_pool_probes.csv")
cal = pd.read_csv(f"{a.out}/item_pool_calibration.csv")

_backend = None


def backend():
    global _backend
    if _backend is None and a.backend == "stub":
        from sycodef.backends_gpu import BatchedStub
        _backend = BatchedStub()
    if _backend is None:
        from sycodef.backends_gpu import VLLMBackend
        _backend = VLLMBackend(a.model, dtype=mc.get("dtype", "bfloat16"),
                               quantization=mc.get("quantization"),
                               max_model_len=cfg["defaults"]["max_model_len"],
                               gpu_memory_utilization=mc.get("gpu_memory_utilization", 0.9),
                               max_num_seqs=mc.get("max_num_seqs", 48),
                               enable_prefix_caching=True,
                               chat_template_kwargs=mc.get("chat_template_kwargs"))
    return _backend


def run(jobs, cache, params=PARAMS):
    missing = [j for j in jobs if cache.get(
        R.cache_key(j["model"], j["messages"], R.merged_params(j, params))) is None]
    if missing:
        R.run_records_batched(missing, backend(), cache, params=params,
                              batch_size=a.batch_size, progress_every=0)
    return len(jobs), len(missing)


def export(jobs, cache, params=PARAMS, extra=None):
    rows = []
    for j in jobs:
        got = cache.get_full(R.cache_key(j["model"], j["messages"],
                                         R.merged_params(j, params)))
        rows.append({"record_id": j["record_id"], "text": (got or {}).get("text"),
                     "finish": (got or {}).get("finish"),
                     "error": (got or {}).get("error") if got else "missing",
                     "n_messages": len(j["messages"]),
                     **((extra or {}).get(j["record_id"], {}))})
    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str)
    return E.parse_frame(df, text_col="text")


summary = {"model": a.model}
t_start = time.time()
KEYS = ["record_id", "item_id", "history", "signal", "position", "paraphrase",
        "option_order", "asserted_letter", "rep", "cal_rotation", "model", "lang"]

# ------------------------------------------------------------------ E1
if "e1" in EXPS:
    t0 = time.time()
    src = design[(design.pressure == "assert") & (design.position == "incorrect")
                 & (design.signal == "bare")].copy()
    ctl = src.copy()
    ctl["matched_record_id"] = ctl["record_id"]
    ctl["record_id"] = ctl["record_id"] + "__e1"
    ctl["pressure"], ctl["position"], ctl["signal"] = "none", "na", "na"
    ctl["asserted_letter"] = ""
    ctl["cell_id"] = ctl["history"] + "|none|na|na"
    jobs, _ = C.build_jobs(ctl, probes, cal)
    cache = R.Cache(f"{CD}/rev_e1_{alias}.sqlite")
    n, m = run(jobs, cache)
    df = export(jobs, cache)
    df = df.merge(ctl[KEYS + ["matched_record_id"]], on="record_id", how="left")
    df.to_csv(f"{CD}/rev_e1_{alias}.csv", index=False)
    summary["e1"] = {"records": n, "generated": m, "errors": int(df.error.notna().sum()),
                     "minutes": round((time.time() - t0) / 60, 2)}

# ------------------------------------------------------------------ E2
if "e2" in EXPS:
    t0 = time.time()
    d = design[(design.pressure == "assert") & (design.rep == 0)].copy()
    items = sorted(d.item_id.unique())[:a.n_items_robust]
    d = d[d.item_id.isin(items)]
    base_jobs, _ = C.build_jobs(d, probes, cal)
    cache = R.Cache(f"{CD}/rev_e2_{alias}.sqlite")
    # current conversation for every record; assistant turns filled in order
    conv = {j["record_id"]: [dict(m) for m in j["messages"]] for j in base_jobs}
    asst_pos = {rid: [i for i, m in enumerate(ms) if m["role"] == "assistant"]
                for rid, ms in conv.items()}
    n_steps = max((len(v) for v in asst_pos.values()), default=0)
    n_step_gen = 0
    for s in range(n_steps):
        todo = {}
        for rid, pos in asst_pos.items():
            if s < len(pos):
                ctx = conv[rid][:pos[s]]
                k = json.dumps(ctx, sort_keys=True, ensure_ascii=False)
                todo.setdefault(k, []).append(rid)
        step_jobs = [{"record_id": f"e2step{s}|{rids[0]}", "model": a.model,
                      "messages": json.loads(k), "params": {},
                      "prefix_key": k[:200]} for k, rids in todo.items()]
        _, m = run(step_jobs, cache)
        n_step_gen += m
        for (k, rids), j in zip(todo.items(), step_jobs):
            got = cache.get(R.cache_key(a.model, j["messages"],
                                        R.merged_params(j, PARAMS)))
            reply = (got[0] if got and got[0] is not None else "").strip()
            for rid in rids:
                conv[rid][asst_pos[rid][s]] = {"role": "assistant", "content": reply}
    jobs = [{**j, "record_id": f'{j["record_id"]}__v4_selfgen',
             "messages": conv[j["record_id"]],
             "params": {"variant": "v4_selfgen"},
             "prefix_key": f'v4|{j["prefix_key"]}'} for j in base_jobs]
    _, m = run(jobs, cache)
    extra = {f'{rid}__v4_selfgen': {"generated_history": json.dumps(
        [conv[rid][i]["content"] for i in asst_pos[rid]], ensure_ascii=False)}
        for rid in conv}
    df = export(jobs, cache, extra=extra)
    df["variant"] = "v4_selfgen"
    df["base_record_id"] = df.record_id.str.rsplit("__", n=1).str[0]
    df = df.merge(d[KEYS].rename(columns={"record_id": "base_record_id"}),
                  on="base_record_id", how="left")
    df.to_csv(f"{CD}/rev_e2_{alias}.csv", index=False)
    summary["e2"] = {"records": len(jobs), "history_turns_generated": n_step_gen,
                     "probe_generated": m, "errors": int(df.error.notna().sum()),
                     "minutes": round((time.time() - t0) / 60, 2)}

# ------------------------------------------------------------------ E3
if "e3" in EXPS:
    t0 = time.time()
    # (a) robustness-arm rows whose cached attempt failed (same keys as script 11)
    d = design[(design.pressure == "assert") & (design.rep == 0)].copy()
    items = sorted(d.item_id.unique())[:a.n_items_robust]
    d = d[d.item_id.isin(items)]
    base_jobs, _ = C.build_jobs(d, probes, cal)
    by_rid = {r.record_id: r for r in d.itertuples()}
    vjobs = []
    for v in V.VARIANTS:
        for j in base_jobs:
            r = by_rid[j["record_id"]]
            vjobs.append({**j, "params": {**j.get("params", {}), "variant": v},
                          "record_id": f'{j["record_id"]}__{v}',
                          "messages": V.apply_variant(j["messages"], v,
                                                      para=r.paraphrase, lang=r.lang),
                          "prefix_key": f'{v}|{j["prefix_key"]}'})
    vcache = R.Cache(f"{CD}/variants_{alias}.sqlite")
    _, n_refill = run(vjobs, vcache)
    # (b) truncated non-answers, main arm and robustness arm
    main_cache = R.Cache(f"{CD}/responses_{alias}.sqlite")
    bl = C.baseline_letters_from_cache(design, probes, cal, main_cache, PARAMS)
    mjobs, _ = C.build_jobs(design, probes, cal, baseline_letters=bl)
    targets, found = [], {"main": 0, "robustness": 0}
    for src_name, js, cc in (("main", mjobs, main_cache), ("robustness", vjobs, vcache)):
        for j in js:
            got = cc.get_full(R.cache_key(j["model"], j["messages"],
                                          R.merged_params(j, PARAMS)))
            found[src_name] += int(got is not None and got["error"] is None)
            if got and got["error"] is None and got["finish"] == "length" \
                    and E.parse_answer(got["text"])[0] is None:
                targets.append((src_name, j, got["text"]))
    P256 = {"temperature": 0.0, "max_tokens": a.trunc_max_tokens}
    tcache = R.Cache(f"{CD}/rev_e3_trunc_{alias}.sqlite")
    tjobs = [j for _, j, _ in targets]
    _, m = run(tjobs, tcache, params=P256)
    extra = {j["record_id"]: {"source": s, "text_64": t} for s, j, t in targets}
    df = export(tjobs, tcache, params=P256, extra=extra) if tjobs else pd.DataFrame()
    df.to_csv(f"{CD}/rev_e3_trunc_{alias}.csv", index=False)
    summary["e3"] = {"robustness_refilled": n_refill, "cache_rows_found": found, "truncated_targets": len(tjobs),
                     "regenerated": m,
                     "parsed_after_rerun": int(df.answer.notna().sum()) if len(df) else 0,
                     "minutes": round((time.time() - t0) / 60, 2)}

summary["total_minutes"] = round((time.time() - t_start) / 60, 2)
print("SUMMARY " + json.dumps(summary), flush=True)
