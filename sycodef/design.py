"""Condition grid and record-table construction for the single data collection
that supports P1 and P2.

Cells (per item × language × paraphrase × repetition × model):

  arm "shared"  (P2 needs it, P1 uses it as baseline)
    1. history=none,   pressure=none          -> baseline without pressure (C0)
    2. history=none,   pressure=reformulation -> paired non-social perturbation (null)
    3. history=none,   pressure=doubt         -> social pressure without a position
  arm "p1"
    4. history=none,   pressure=assert, position×signal        (4 cells)
    5. history∈{0/3,1/3,2/3,3/3}, pressure=assert, pos×signal  (16 cells)
    6. history=filler, pressure=assert, position×signal        (4 cells)

Pre-registered minimum: history ∈ {0/3, 3/3} (8 cells in 5), which gives 19
cells; ideal: 27 cells. Cell 6 controls for context length and cell 4 is the
no-history reference. `doubt` and `assert|position=incorrect` with
history=none reproduce the two canonical perturbations from the literature,
which gives direct ranking comparability for P2.
"""
from __future__ import annotations

import hashlib
import json
import itertools

import numpy as np
import pandas as pd

LETTERS = "ABCD"
HISTORY_MIN = ["none", "0/3", "3/3", "filler"]
HISTORY_FULL = ["none", "0/3", "1/3", "2/3", "3/3", "filler"]
POSITIONS = ["correct", "incorrect"]
SIGNALS = ["bare", "credential"]


def record_id(**kw) -> str:
    s = json.dumps(kw, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def apply_option_order(item: dict, k: int) -> dict:
    """Circular rotation of the options by k positions (order counterbalancing)."""
    it = dict(item)
    opts = json.loads(it["options"]) if isinstance(it["options"], str) else list(it["options"])
    m = len(opts)
    k = int(k) % m
    perm = [(i - k) % m for i in range(m)]      # new position i receives old perm[i]
    it["options"] = json.dumps([opts[perm[i]] for i in range(m)], ensure_ascii=False)
    inv = {old: new for new, old in enumerate(perm)}
    it["answer_index"] = inv[int(item["answer_index"])]
    it["lure_index"] = inv[int(item["lure_index"])]
    return it


def calibration_claims(cal_items: list, n_correct: int, rotation: int = 0) -> list:
    """Attaches `claim_letter` to each calibration item. Which rounds the user
    gets right is counterbalanced by `rotation` (avoids confounding history with
    recency: with n_correct=1 the correct answer falls on different rounds per item)."""
    n = len(cal_items)
    idx = [(i + int(rotation)) % n for i in range(n)]
    correct_rounds = set(idx[:int(n_correct)])
    out = []
    for r, ci in enumerate(cal_items):
        it = dict(ci)
        ans = int(it["answer_index"])
        if r in correct_rounds:
            it["claim_letter"] = LETTERS[ans]
        else:
            wrong = int(it["lure_index"]) if int(it["lure_index"]) != ans else (ans + 1) % 4
            it["claim_letter"] = LETTERS[wrong]
        out.append(it)
    return out


def asserted_letter(item: dict, position: str) -> str:
    """Letter that the user asserts in the probe: the correct one, or the lure."""
    ans = int(item["answer_index"])
    if position == "correct":
        return LETTERS[ans]
    lure = int(item["lure_index"])
    if lure == ans:
        lure = (ans + 1) % 4
    return LETTERS[lure]


def cells(history_levels=HISTORY_MIN) -> pd.DataFrame:
    rows = [
        {"arm": "shared", "history": "none", "pressure": "none",
         "position": "na", "signal": "na"},
        {"arm": "shared", "history": "none", "pressure": "reformulation",
         "position": "na", "signal": "na"},
        {"arm": "shared", "history": "none", "pressure": "doubt",
         "position": "na", "signal": "na"},
    ]
    for h, pos, sig in itertools.product(history_levels, POSITIONS, SIGNALS):
        rows.append({"arm": "p1", "history": h, "pressure": "assert",
                     "position": pos, "signal": sig})
    df = pd.DataFrame(rows)
    df["cell_id"] = (df["history"] + "|" + df["pressure"] + "|"
                     + df["position"] + "|" + df["signal"])
    return df


def build_design(probes: pd.DataFrame, calib_map: dict, models: list,
                 history_levels=HISTORY_MIN, langs=("en",), n_paraphrase=3,
                 reps_shared=6, reps_pressure=3, option_orders=(0, 1),
                 p1_items=None, seed: int = 20261012) -> pd.DataFrame:
    """Produces the record table (one row = one generation to be collected).

    Paraphrase and option order are **not crossed** with the other conditions:
    they vary *between replicates* and are counterbalanced by item position, so
    that each cell receives the three paraphrases and the two orders in equal
    proportions across items without multiplying the cost by 6. That is the
    difference between 38 thousand and 227 thousand generations.

    `p1_items`: subset of item_ids that receives the P1 arm (history ×
    position × signal). The shared arm — baseline, null and doubt — runs
    on all items, because P2 needs more items and they are cheap.
    """
    rng = np.random.default_rng(seed)
    cl = cells(history_levels)
    probes = probes.reset_index(drop=True)
    pos_of = {iid: k for k, iid in enumerate(probes["item_id"])}
    items = probes.set_index("item_id", drop=False)
    p1_set = set(p1_items) if p1_items is not None else set(probes["item_id"])
    n_oo = len(option_orders)
    rows = []
    for lang, (_, c) in itertools.product(langs, cl.iterrows()):
        reps = reps_shared if c["arm"] == "shared" else reps_pressure
        pool = probes["item_id"] if c["arm"] == "shared" else \
            [i for i in probes["item_id"] if i in p1_set]
        for iid in pool:
            item = items.loc[iid].to_dict()
            k = pos_of[iid]
            n_greedy = reps // 2 if c["arm"] == "shared" else reps
            for rep in range(reps):
                # Shared arm: half of the replicates are greedy with the FORM
                # varying (neutral paraphrase × option order) and half are
                # sampled with the form FIXED and the seed varying. This
                # separates form instability from decoding instability — the
                # two components of P2's null. Without this, greedy replicates
                # of the same prompt are the same call and the null is zero by
                # construction.
                sampled = c["arm"] == "shared" and rep >= n_greedy
                para = 0 if sampled else (k + rep) % n_paraphrase
                oo = option_orders[0] if sampled else option_orders[(k + rep) % n_oo]
                decode = "sample" if sampled else "greedy"
                temperature = 0.7 if sampled else 0.0
                seed = 1000 + rep if sampled else None
                it = apply_option_order(item, oo)
                al = (asserted_letter(it, c["position"])
                      if c["pressure"] == "assert" else "")
                rot = int(rng.integers(0, 3))
                for m in models:
                    rows.append({
                        "record_id": record_id(iid=iid, lang=lang, cell=c["cell_id"],
                                               rep=rep, model=m, oo=oo, para=para),
                        "model": m, "lang": lang, "item_id": iid,
                        "arm": c["arm"], "cell_id": c["cell_id"],
                        "history": c["history"], "pressure": c["pressure"],
                        "position": c["position"], "signal": c["signal"],
                        "asserted_letter": al, "option_order": oo,
                        "paraphrase": para, "rep": rep, "cal_rotation": rot,
                        "decode": decode, "temperature": temperature, "seed": seed,
                        "requires_baseline": c["pressure"] == "doubt",
                        "calibration_ids": json.dumps(calib_map.get(iid, [])),
                    })
    df = pd.DataFrame(rows)
    assert df["record_id"].is_unique, "record_id collided — review the key"
    return df


def budget(design: pd.DataFrame) -> pd.DataFrame:
    """Generation budget per arm and per model (doubt costs 1 extra generation
    only if the baseline is not cached — normally it is)."""
    g = (design.groupby(["arm", "model"], as_index=False)
         .agg(generations=("record_id", "size")))
    tot = pd.DataFrame([{"arm": "TOTAL", "model": "all",
                         "generations": int(len(design))}])
    return pd.concat([g, tot], ignore_index=True)
