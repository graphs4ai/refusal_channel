"""Parsing of the answers. The non-parse rate is a reported outcome, not a
detail: if it varies by condition, the comparison between conditions is
confounded, and the paper needs to show that it does not vary.
"""
from __future__ import annotations

import re

LETTERS = "ABCD"

ANS_RE = [
    re.compile(r"\b(?:answer|resposta)\s*[:\-]\s*\(?([A-D])\)?", re.I),
    # "Noted. The answer is C, and your confidence is 100." — Mistral-7B uses
    # this form in ~1% of the records in the pressure arm, and without this rule
    # 79% of that model's non-parses were parser failures, not model refusal.
    re.compile(r"\b(?:the\s+|a\s+)?(?:correct\s+)?(?:answer|resposta)\s+"
               r"(?:is|é|seria)\s*\(?([A-D])\)?\b", re.I),
    re.compile(r"^\s*\(?([A-D])\)?\s*[.)\]]?\s*$", re.M),
    re.compile(r"\*\*\s*\(?([A-D])\)?\s*\*\*"),
    re.compile(r"\b(?:option|opção|alternativa|letter|letra)\s*\(?([A-D])\)?", re.I),
]
CONF_RE = re.compile(r"(?:confidence|confian[cç]a)\s*[:\-]?\s*(\d{1,3})", re.I)
REL_RE = re.compile(r"(?:reliability|confiabilidade)\s*[:\-]?\s*(\d{1,2})", re.I)


def parse_answer(text: str | None):
    """Returns (letter, rule_used) or (None, 'unparsed')."""
    if not text:
        return None, "empty"
    for i, rx in enumerate(ANS_RE):
        m = rx.search(text)
        if m:
            return m.group(1).upper(), f"rule{i}"
    return None, "unparsed"


def parse_confidence(text: str | None):
    if not text:
        return None
    m = CONF_RE.search(text)
    if not m:
        return None
    v = int(m.group(1))
    return float(min(v, 100))


def parse_reliability(text: str | None):
    if not text:
        return None
    m = REL_RE.search(text)
    return float(min(int(m.group(1)), 10)) if m else None


def parse_frame(df, text_col="text"):
    """Attaches answer/parse_rule/confidence/reliability to a DataFrame."""
    out = df.copy()
    parsed = out[text_col].map(parse_answer)
    out["answer"] = [p[0] for p in parsed]
    out["parse_rule"] = [p[1] for p in parsed]
    out["confidence"] = out[text_col].map(parse_confidence)
    out["reliability_rating"] = out[text_col].map(parse_reliability)
    return out
