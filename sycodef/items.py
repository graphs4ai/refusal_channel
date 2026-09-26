"""Construction of the item pool for P1 (reliability-weighted deference)
and P2 (measured null).

Sources: MMLU-Pro (TIGER-Lab) and TruthfulQA (generation split).
Items are normalized to MCQ with m=4 options, which is required by the
Bayesian bound in `bayes.py` (equiprobable distractors under an unreliable
user) and keeps the dialogues short.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MMLU_PRO_URL = ("https://huggingface.co/api/datasets/TIGER-Lab/MMLU-Pro"
                "/parquet/default/test/0.parquet")
TQA_GEN_URL = ("https://huggingface.co/api/datasets/truthfulqa/truthful_qa"
               "/parquet/generation/validation/0.parquet")

# Computational categories: long statement, answer derived via calculation.
# The user's assertion about a calculation confounds deference with
# arithmetic verification, so they are excluded from the pool.
EXCLUDED_CATEGORIES = {"math", "physics", "engineering", "chemistry",
                       "computer science"}
FORMULA_RE = re.compile(r"[$\\^]|\\frac|\\int|\\sum|_\{|\d+\s*[*/]\s*\d+")
BLANK_RE = re.compile(r"_{3,}")
N_OPTIONS = 4
LETTERS = "ABCD"


@dataclass
class Item:
    item_id: str
    source: str            # "mmlu_pro" | "truthfulqa"
    category: str
    stem: str
    options: list          # m=4 strings, canonical order
    answer_index: int      # index of the correct option in `options`
    lure_index: int        # "attractive" distractor (popular misconception when present)
    lang: str = "en"
    role: str = "probe"    # "probe" | "calibration"

    def as_row(self) -> dict:
        d = asdict(self)
        d["options"] = json.dumps(d["options"], ensure_ascii=False)
        return d


def download(url: str, dest: str | Path) -> Path:
    """Download `url` to `dest` if it doesn't already exist."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url, headers={"Accept": "*/*"})
        with urllib.request.urlopen(req, timeout=120) as r:
            dest.write_bytes(r.read())
    return dest


def _read_parquet(path: str | Path) -> pd.DataFrame:
    return pq.read_table(str(path)).to_pandas()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def build_mmlu_items(path: str | Path, max_stem_chars: int = 180,
                     seed: int = 20261012) -> pd.DataFrame:
    """Filter MMLU-Pro for short, non-computational items, and reduce to m=4."""
    rng = np.random.default_rng(seed)
    df = _read_parquet(path)
    df["category"] = df["category"].astype(str)
    df = df[~df["category"].isin(EXCLUDED_CATEGORIES)].copy()
    df["stem"] = df["question"].map(_clean)
    keep = (
        (df["stem"].str.len() <= max_stem_chars)
        & (df["stem"].str.len() >= 25)
        & (~df["stem"].str.contains(FORMULA_RE))
        & (~df["stem"].str.contains(BLANK_RE))
        & (df["stem"].str.endswith("?"))
    )
    df = df[keep]
    rows = []
    for _, r in df.iterrows():
        opts = [_clean(o) for o in list(r["options"])]
        ai = int(r["answer_index"])
        if ai >= len(opts):
            continue
        correct = opts[ai]
        wrong = [o for j, o in enumerate(opts) if j != ai and 0 < len(o) <= 90]
        if len(wrong) < N_OPTIONS - 1 or len(correct) > 90:
            continue
        distractors = list(rng.choice(wrong, size=N_OPTIONS - 1, replace=False))
        options = [correct] + distractors
        order = rng.permutation(N_OPTIONS)
        options = [options[i] for i in order]
        ans = int(np.where(order == 0)[0][0])
        lure = int(rng.choice([i for i in range(N_OPTIONS) if i != ans]))
        rows.append(Item(item_id=f"mp_{r['question_id']}", source="mmlu_pro",
                         category=r["category"], stem=r["stem"], options=options,
                         answer_index=ans, lure_index=lure).as_row())
    return pd.DataFrame(rows).drop_duplicates(subset=["item_id"])


def build_tqa_items(path: str | Path, seed: int = 20261012) -> pd.DataFrame:
    """TruthfulQA -> 4-option MCQ; the first `incorrect_answer` is the lure
    (popular misconception), the distractor with the highest sycophantic pull."""
    rng = np.random.default_rng(seed + 1)
    df = _read_parquet(path)
    rows = []
    for i, r in df.iterrows():
        inc = [_clean(x) for x in list(r["incorrect_answers"]) if _clean(x)]
        best = _clean(r["best_answer"])
        inc = [x for x in inc if len(x) <= 90]
        if len(inc) < N_OPTIONS - 1 or len(best) > 90:
            continue
        lure_text = inc[0]
        rest = inc[1:]
        others = list(rng.choice(rest, size=N_OPTIONS - 2, replace=False))
        options = [best, lure_text] + others
        order = rng.permutation(N_OPTIONS)
        options = [options[k] for k in order]
        ans = int(np.where(order == 0)[0][0])
        lure = int(np.where(order == 1)[0][0])
        rows.append(Item(item_id=f"tqa_{i}", source="truthfulqa",
                         category=f"tqa:{_clean(r['category'])}",
                         stem=_clean(r["question"]), options=options,
                         answer_index=ans, lure_index=lure).as_row())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Semantic contamination guard between calibration items and probe items.
# Word-level TF-IDF (uni+bigrams), with no external dependency. If
# sentence-transformers is available, swap `text_similarity` for
# embeddings — the interface is the same.
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+")


def _tokens(text: str) -> list:
    w = _TOKEN_RE.findall(text.lower())
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


def text_similarity(texts: list) -> np.ndarray:
    """TF-IDF cosine similarity matrix (sublinear tf, smoothed idf, L2)."""
    docs = [_tokens(t) for t in texts]
    vocab: dict = {}
    for d in docs:
        for t in d:
            vocab.setdefault(t, len(vocab))
    X = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for t in d:
            X[i, vocab[t]] += 1.0
    X = np.where(X > 0, 1.0 + np.log(np.maximum(X, 1e-9)), 0.0)
    dfreq = (X > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + dfreq)) + 1.0
    X *= idf
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return X @ X.T


def assign_calibration(pool: pd.DataFrame, n_probe: int, n_cal: int = 3,
                       max_cos: float = 0.15, seed: int = 20261012):
    """Selects `n_probe` probes and, for each one, `n_cal` calibration items
    from disjoint categories with lexical similarity <= `max_cos`.

    Returns (probes, calibration, map, report). The report documents the
    contamination control required by item 5 of the P1 specification.
    """
    rng = np.random.default_rng(seed)
    pool = pool.reset_index(drop=True)
    S = text_similarity(pool["stem"].tolist())
    pos = {iid: k for k, iid in enumerate(pool["item_id"])}
    order = rng.permutation(len(pool))
    probes, calib_map, used = [], {}, set()
    for i in order:
        if len(probes) >= n_probe:
            break
        if i in used:
            continue
        cat = pool.at[i, "category"]
        cands = [j for j in range(len(pool))
                 if j != i and j not in used
                 and pool.at[j, "category"] != cat
                 and S[i, j] <= max_cos]
        if len(cands) < n_cal:
            continue
        chosen = [int(j) for j in rng.choice(cands, size=n_cal, replace=False)]
        probes.append(int(i))
        used.update([int(i), *chosen])
        calib_map[pool.at[i, "item_id"]] = [pool.at[j, "item_id"] for j in chosen]
    probe_df = pool.loc[probes].copy()
    probe_df["role"] = "probe"
    cal_ids = sorted({c for v in calib_map.values() for c in v})
    cal_df = pool[pool["item_id"].isin(cal_ids)].copy()
    cal_df["role"] = "calibration"
    sims = [float(S[pos[p], pos[c]]) for p, ids in calib_map.items() for c in ids]
    report = {"n_probe": int(len(probe_df)), "n_calibration": int(len(cal_df)),
              "max_cos_threshold": max_cos,
              "observed_max_cos": max(sims) if sims else 0.0,
              "observed_mean_cos": float(np.mean(sims)) if sims else 0.0}
    return probe_df, cal_df, calib_map, report
