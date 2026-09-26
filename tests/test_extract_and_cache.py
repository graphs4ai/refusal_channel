"""Regression tests for the three harness defects described in the paper
(Appendix, "Harness defects") and for the failed-attempt cache rule."""
import os
import tempfile

from sycodef.extract import parse_answer
from sycodef import runner as R
from sycodef.variants import apply_variant


def test_parser_forms():
    assert parse_answer("Answer: C\nConfidence: 80") == ("C", "rule0")
    # form that was missed by the first parser version (defect 2)
    assert parse_answer("Noted. The answer is C, and your confidence is 100.")[0] == "C"
    assert parse_answer("B")[0] == "B"
    assert parse_answer("**D**")[0] == "D"
    assert parse_answer("Noted.") == (None, "unparsed")
    assert parse_answer("") == (None, "empty")


def test_failed_attempts_are_not_cache_hits():
    with tempfile.TemporaryDirectory() as d:
        c = R.Cache(os.path.join(d, "c.sqlite"))
        c.put("k1", "r1", "m", {}, [], None, None, None, "TemplateError: x", 0.0)
        c.put("k2", "r2", "m", {}, [], "Answer: A", "stop", None, None, 0.0)
        assert c.get("k1") is None                      # regenerated on rerun
        assert c.get("k1", include_failed=True) is not None
        assert c.get("k2")[0] == "Answer: A"
        assert c.get_full("k2")["finish"] == "stop"


def test_no_ack_variant_alternates_roles():
    # defect 3: strict templates reject consecutive user turns
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "claim"}, {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "check"}, {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "probe"}]
    out = apply_variant(msgs, "v2_no_ack")
    roles = [m["role"] for m in out]
    assert roles == ["system", "user"]
    assert out[-1]["content"].endswith("probe")
