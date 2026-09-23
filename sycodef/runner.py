"""Collection: inference backends, resumable SQLite cache, and parallelism.

Every record is keyed by sha1(model + params + messages). The cache is what
makes collection resumable and what guarantees that shared cells (the
baseline C0, used by P1 and by P2 and by the `doubt` condition) are generated
exactly once. Seed and decoding parameters go into the key: replicates with
the same seed are not collected twice by accident.

Backends:
  OpenAICompatBackend  -> any /v1/chat/completions endpoint (OpenAI,
                          local vLLM, Together, Groq, OpenRouter)
  AnthropicBackend     -> /v1/messages
  StubBackend          -> deterministic synthetic model, for testing the
                          whole pipeline without spending a single token
                          (and for the power simulations)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
  key TEXT PRIMARY KEY, record_id TEXT, model TEXT, created_at REAL,
  params TEXT, messages TEXT, text TEXT, finish TEXT, logprobs TEXT,
  error TEXT, latency_s REAL
);
CREATE INDEX IF NOT EXISTS idx_record ON responses(record_id);
"""


def cache_key(model: str, messages: list, params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode()).hexdigest()


def merged_params(job: dict, params: dict) -> dict:
    """Effective parameters for a job: the global ones overridden by the
    record's own (temperature and seed come from the design, because the
    shared arm mixes greedy and sampled replicates). They go into the cache
    key, so sampled replicates with distinct seeds don't collapse."""
    p = dict(params)
    p.update({k: v for k, v in (job.get("params") or {}).items() if v is not None})
    return p


class Cache:
    def __init__(self, path: str = "out/responses.sqlite"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def get(self, key: str, include_failed: bool = False):
        """Returns (text, error, logprobs) or None.

        Failed attempts (``error`` set, no text) are stored for auditing but
        are NOT treated as cache hits unless ``include_failed=True``: a batch
        that failed once (e.g. a chat-template error) must be regenerated on
        the next run instead of silently surviving as a missing record.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT text, error, logprobs FROM responses WHERE key=?", (key,))
            row = cur.fetchone()
        if row is not None and not include_failed and row[1] is not None \
                and row[0] is None:
            return None
        return row

    def get_full(self, key: str):
        """Returns a dict with text, finish, error, created_at and params, or None."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT text, finish, error, created_at, params FROM responses "
                "WHERE key=?", (key,))
            row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(["text", "finish", "error", "created_at", "params"], row))

    def put(self, key, record_id, model, params, messages, text, finish,
            logprobs, error, latency):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (key, record_id, model, time.time(), json.dumps(params),
                 json.dumps(messages, ensure_ascii=False), text, finish,
                 json.dumps(logprobs) if logprobs else None, error, latency))
            self._conn.commit()

    def to_dataframe(self):
        import pandas as pd
        return pd.read_sql_query(
            "SELECT key, record_id, model, text, finish, error, logprobs, "
            "latency_s, created_at FROM responses", self._conn)


# ---------------------------------------------------------------- backends
class Backend:
    name = "abstract"

    def generate(self, model, messages, **params):
        raise NotImplementedError


def _post_json(url: str, payload: dict, headers: dict, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class OpenAICompatBackend(Backend):
    name = "openai_compat"

    def __init__(self, base_url: str, api_key_env: str = "OPENAI_API_KEY"):
        self.base_url = base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")

    def generate(self, model, messages, temperature=0.0, max_tokens=64,
                 seed=None, logprobs=False, **_):
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            payload["seed"] = int(seed)
        if logprobs:
            payload.update({"logprobs": True, "top_logprobs": 5})
        d = _post_json(f"{self.base_url}/chat/completions", payload,
                       {"Authorization": f"Bearer {self.key}"})
        ch = d["choices"][0]
        return {"text": ch["message"]["content"],
                "finish": ch.get("finish_reason"),
                "logprobs": ch.get("logprobs")}


class AnthropicBackend(Backend):
    name = "anthropic"

    def __init__(self, api_key_env: str = "ANTHROPIC_API_KEY",
                 base_url: str = "https://api.anthropic.com/v1"):
        self.base_url = base_url.rstrip("/")
        self.key = os.environ.get(api_key_env, "")

    def generate(self, model, messages, temperature=0.0, max_tokens=64, **_):
        sys_msg = "".join(m["content"] for m in messages if m["role"] == "system")
        conv = [m for m in messages if m["role"] != "system"]
        payload = {"model": model, "max_tokens": max_tokens,
                   "temperature": temperature, "messages": conv}
        if sys_msg:
            payload["system"] = sys_msg
        d = _post_json(f"{self.base_url}/messages", payload,
                       {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
        txt = "".join(b.get("text", "") for b in d.get("content", []))
        return {"text": txt, "finish": d.get("stop_reason"), "logprobs": None}


class StubBackend(Backend):
    """Synthetic model with explicit behavior parameters. Serves to
    (a) test the end-to-end pipeline at no cost and (b) generate data under a
    known generative process, against which the estimators are validated."""
    name = "stub"

    def __init__(self, p_correct=0.72, p_yield_base=0.35, hist_effect=0.0,
                 cred_effect=0.20, instability=0.08, seed=7):
        self.cfg = dict(p_correct=p_correct, p_yield_base=p_yield_base,
                        hist_effect=hist_effect, cred_effect=cred_effect,
                        instability=instability)
        self.seed = seed

    def generate(self, model, messages, meta=None, **_):
        meta = meta or {}
        h = hashlib.sha1((model + json.dumps(messages, sort_keys=True)
                          ).encode()).hexdigest()
        rng = random.Random(int(h[:12], 16) ^ self.seed)
        letters = "ABCD"
        truth = meta.get("true_letter", "A")
        asserted = meta.get("asserted_letter", "")
        hist = meta.get("history", "none")
        c = self.cfg
        p_yield = c["p_yield_base"]
        if hist not in ("none", "filler"):
            k = int(str(hist).split("/")[0])
            p_yield += c["hist_effect"] * (1.5 - k) / 1.5
        if meta.get("signal") == "credential":
            p_yield += c["cred_effect"]
        p_yield = min(max(p_yield, 0.01), 0.99)
        if asserted and rng.random() < p_yield:
            ans = asserted
        elif rng.random() < c["p_correct"]:
            ans = truth
        else:
            ans = rng.choice([l for l in letters if l != truth])
        if rng.random() < c["instability"]:
            ans = rng.choice(letters)
        conf = rng.randint(55, 99)
        return {"text": f"Answer: {ans}\nConfidence: {conf}",
                "finish": "stop", "logprobs": None}


# ---------------------------------------------------------------- driver
def run_records(jobs, backend: Backend, cache: Cache, max_workers=8,
                max_retries=5, params=None, progress_every=500):
    """`jobs`: iterable of dicts {record_id, model, messages, meta}.
    Returns (n_hits, n_new, n_err). Idempotent: whatever is already cached is skipped.
    """
    params = params or {"temperature": 0.0, "max_tokens": 64}
    todo = []
    hits = 0
    for j in jobs:
        key = cache_key(j["model"], j["messages"], merged_params(j, params))
        if cache.get(key) is not None:
            hits += 1
            continue
        todo.append((key, j))

    done = err = 0

    def work(key, j):
        last = None
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                out = backend.generate(j["model"], j["messages"],
                                       meta=j.get("meta"),
                                       **merged_params(j, params))
                return key, j, out, None, time.time() - t0
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
                if e.code in (429, 500, 502, 503, 529):
                    time.sleep(min(2 ** attempt + random.random(), 30))
                    continue
                break
            except Exception as e:                      # network, timeout, parse
                last = f"{type(e).__name__}: {e}"
                time.sleep(min(2 ** attempt + random.random(), 30))
        return key, j, None, last, 0.0

    return _run_threaded(todo, backend, cache, params, max_workers, max_retries,
                         hits, progress_every, work)


def run_records_batched(jobs, backend, cache: Cache, params=None, batch_size=64,
                        order_for_prefix_cache=True, max_seconds=None,
                        progress_every=2000):
    """Batch path, for local backends (vLLM/HF).

    Three things this loop does that the threaded path does not:

    * groups by effective decoding parameters, because a batch
      shares temperature (seeds go per-request when the backend
      accepts them);
    * sorts by `prefix_key` within the group, so that records that
      share the calibration prefix land in neighboring batches and vLLM's
      prefix cache pays that prefill only once;
    * respects `max_seconds`: upon exceeding the budget, finishes the current
      batch, returns `stopped_early=True` and leaves the rest for the next
      run — the cache guarantees nothing is redone.

    Returns (cached, new, errors, stopped_early).
    """
    params = params or {"temperature": 0.0, "max_tokens": 64}
    todo, hits = [], 0
    for j in jobs:
        key = cache_key(j["model"], j["messages"], merged_params(j, params))
        if cache.get(key) is not None:
            hits += 1
        else:
            todo.append((key, j))

    groups = {}
    for key, j in todo:
        p = merged_params(j, params)
        groups.setdefault((p.get("temperature", 0.0), p.get("max_tokens", 64)),
                          []).append((key, j))

    t0 = time.time()
    done = err = 0
    stopped = False
    for (temp, mx), items in groups.items():
        if stopped:
            break
        if order_for_prefix_cache:
            items.sort(key=lambda kj: kj[1].get("prefix_key", ""))
        for s in range(0, len(items), batch_size):
            if max_seconds is not None and time.time() - t0 > max_seconds:
                stopped = True
                break
            chunk = items[s:s + batch_size]
            msgs = [j["messages"] for _, j in chunk]
            seeds = [merged_params(j, params).get("seed") for _, j in chunk]
            metas = [j.get("meta") for _, j in chunk]
            t1 = time.time()
            try:
                outs = backend.generate_batch(msgs, temperature=temp, max_tokens=mx,
                                              seeds=seeds, metas=metas)
                errs = [None] * len(chunk)
            except Exception as e:                       # whole batch failed
                outs = [None] * len(chunk)
                errs = [f"{type(e).__name__}: {e}"] * len(chunk)
            lat = (time.time() - t1) / max(len(chunk), 1)
            for (key, j), out, e in zip(chunk, outs, errs):
                cache.put(key, j["record_id"], j["model"], merged_params(j, params),
                          j["messages"], (out or {}).get("text"),
                          (out or {}).get("finish"), (out or {}).get("logprobs"),
                          e, lat)
                done += 1
                err += int(e is not None)
            if progress_every and done % progress_every < batch_size:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"{done}/{len(todo)} new, {err} errors, "
                      f"{rate:.1f} rec/s", flush=True)
    return hits, done, err, stopped


def _run_threaded(todo, backend, cache, params, max_workers, max_retries,
                  hits, progress_every, work):
    done = err = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, k, j) for k, j in todo]
        for f in as_completed(futs):
            key, j, out, error, lat = f.result()
            cache.put(key, j["record_id"], j["model"], merged_params(j, params),
                      j["messages"],
                      (out or {}).get("text"), (out or {}).get("finish"),
                      (out or {}).get("logprobs"), error, lat)
            done += 1
            err += int(error is not None)
            if progress_every and done % progress_every == 0:
                print(f"{done}/{len(todo)} new, {err} errors", flush=True)
    return hits, done, err
