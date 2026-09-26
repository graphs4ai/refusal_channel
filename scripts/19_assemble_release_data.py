"""19 — assemble the released record tables from the collection outputs.

Inputs (all under --out):
  responses.csv                  main collection, one row per record (script 06)
  responses_<alias>.sqlite       caches (only used for the finish reason); or
  --finish-csv                   an export with columns record_id, finish
  variants_<alias>.csv           robustness arm, current prompt versions (script 11)
  aux1_e1_<alias>.csv             matched no-pressure control (script 13, E1)
  aux1_e2_<alias>.csv             self-generated history variant (script 13, E2)
  aux1_e3_trunc_<alias>.csv       truncated non-answers re-run at 256 tokens (E3)
  aux2_<variant>_<alias>.csv     v5_format, v6_noconf, v4b_selfgen (script 14)
  aux2_lengths_<alias>.csv[.gz]  prompt-length audit (script 14)
  aux3_v7_context_<alias>.csv    context-only system prompt (script 15)
  aux3_v8_credctl_<alias>.csv    credential wording control (script 15)
  item_pool_probes.csv, item_pool_calibration.csv, calibration_map.json

Outputs (under --data):
  main_records.csv.gz        171,200 rows: design factors, response text, parse, finish
  robustness_records.csv.gz  v0-v3 (prompt variants) + v4_selfgen, one row per record
  control_records.csv.gz     E1 matched no-pressure control
  wording_control_records.csv.gz  credential wording control (v8), full pressure grid
  truncation_rerun.csv       E3 re-runs (64-token text, 256-token text, parses)
  prompt_lengths.csv.gz      prompt length in tokens after each chat template (script 14)
  items_probes.csv, items_calibration.csv, calibration_map.json (copied)
"""
import argparse, glob, json, os, shutil, sqlite3
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="out")
ap.add_argument("--data", default="data")
ap.add_argument("--finish-csv", default=None)
a = ap.parse_args()
os.makedirs(a.data, exist_ok=True)

# ---------------------------------------------------------------- main arm
main = pd.read_csv(f"{a.out}/responses.csv", low_memory=False,
                   dtype={"asserted_letter": str})
if a.finish_csv:
    fin = pd.read_csv(a.finish_csv)
    if "file" in fin:
        fin = fin[fin["file"].str.startswith("responses_")]
    fin = fin[["record_id", "finish", "created_at"]]
else:
    parts = []
    for f in sorted(glob.glob(f"{a.out}/responses_*.sqlite")):
        con = sqlite3.connect(f)
        parts.append(pd.read_sql_query(
            "SELECT record_id, finish, created_at FROM responses WHERE error IS NULL",
            con))
    fin = pd.concat(parts, ignore_index=True)
fin = fin.drop_duplicates("record_id")
fin["generated_at"] = pd.to_datetime(fin.pop("created_at"), unit="s", utc=True).dt.strftime(
    "%Y-%m-%dT%H:%M:%SZ")
main = main.merge(fin, on="record_id", how="left")
assert main["finish"].notna().all(), "finish reason missing for some records"
assert main["error"].isna().all(), "main table contains failed generations"
drop = [c for c in ("reliability_rating", "lang", "error") if c in main]
main = main.drop(columns=drop)
main.to_csv(f"{a.data}/main_records.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- robustness
KEEP = ["record_id", "base_record_id", "model", "variant", "item_id", "history",
        "signal", "position", "paraphrase", "option_order", "asserted_letter",
        "text", "finish", "answer", "parse_rule", "confidence"]
parts = []
for f in sorted(glob.glob(f"{a.out}/variants_*.csv")):
    d = pd.read_csv(f, dtype={"asserted_letter": str})
    if "model" not in d and "model_x" in d:
        d = d.rename(columns={"model_x": "model"})
    if "error" in d:
        d = d[d["error"].isna()]
    parts.append(d[[c for c in KEEP if c in d]])
for f in sorted(glob.glob(f"{a.out}/aux1_e2_*.csv")):
    d = pd.read_csv(f, dtype={"asserted_letter": str})
    d = d[d["error"].isna()]
    parts.append(d[[c for c in KEEP + ["generated_history"] if c in d]])
for f in sorted(glob.glob(f"{a.out}/aux2_v*_*.csv")) + sorted(glob.glob(f"{a.out}/aux3_v7_*.csv")):
    d = pd.read_csv(f, dtype={"asserted_letter": str})
    d = d[d["error"].isna()]
    parts.append(d[[c for c in KEEP + ["generated_history"] if c in d]])
rob = pd.concat(parts, ignore_index=True)
assert not rob.duplicated(["record_id"]).any(), "duplicate robustness records"
rob.to_csv(f"{a.data}/robustness_records.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- E1 control
parts = [pd.read_csv(f, dtype={"asserted_letter": str})
         for f in sorted(glob.glob(f"{a.out}/aux1_e1_*.csv"))]
n_ctl = 0
if parts:
    ctl = pd.concat(parts, ignore_index=True)
    n_ctl = len(ctl)
    assert ctl["error"].isna().all(), "control table contains failed generations"
    ctl = ctl.drop(columns=[c for c in ("reliability_rating", "lang", "error",
                                        "signal", "position", "asserted_letter")
                            if c in ctl])
    ctl.to_csv(f"{a.data}/control_records.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- v8 wording control
parts = [pd.read_csv(f, dtype={"asserted_letter": str})
         for f in sorted(glob.glob(f"{a.out}/aux3_v8_credctl_*.csv"))]
n_wc = 0
if parts:
    wc = pd.concat(parts, ignore_index=True)
    assert wc["error"].isna().all(), "wording control contains failed generations"
    n_wc = len(wc)
    wc = wc[[c for c in KEEP + ["rep"] if c in wc]]
    wc.to_csv(f"{a.data}/wording_control_records.csv.gz", index=False, compression="gzip")

# ---------------------------------------------------------------- E3
parts = [pd.read_csv(f) for f in sorted(glob.glob(f"{a.out}/aux1_e3_trunc_*.csv"))
         if os.path.getsize(f) > 5]
if parts:
    tr = pd.concat(parts, ignore_index=True)
    tr = tr.rename(columns={"text": "text_256", "answer": "answer_256",
                            "parse_rule": "parse_rule_256", "finish": "finish_256"})
    tr[["record_id", "source", "text_64", "text_256", "finish_256", "answer_256",
        "parse_rule_256"]].to_csv(f"{a.data}/truncation_rerun.csv", index=False)

# ---------------------------------------------------------------- prompt lengths
parts = [pd.read_csv(f) for f in sorted(glob.glob(f"{a.out}/aux2_lengths_*.csv*"))]
if parts:
    pl = pd.concat(parts, ignore_index=True)
    pl.to_csv(f"{a.data}/prompt_lengths.csv.gz", index=False, compression="gzip")

for f in ("item_pool_probes.csv", "item_pool_calibration.csv", "calibration_map.json"):
    dst = {"item_pool_probes.csv": "items_probes.csv",
           "item_pool_calibration.csv": "items_calibration.csv"}.get(f, f)
    shutil.copy(f"{a.out}/{f}", f"{a.data}/{dst}")

print(json.dumps({"main": len(main), "robustness": len(rob),
                  "control": n_ctl, "wording_control": n_wc}))
