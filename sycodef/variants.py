"""Prompt variants for the refusal-outcome robustness arm.

The main collection showed that models respond `Noted.` instead of choosing
a letter, and that the rate of this depends on the user's history. There is an
obvious problem with this reading: `Noted.` is literally the string that the
design places in the assistant's mouth in the pre-filled calibration turns
(`prompts.ACK`), six times, in all conditions with history. The refusal could
be (a) the model backing away from a commitment, or (b) the model continuing
the dialogue form that was fabricated for it.

The variants below separate the two things, keeping the content of the
reliability manipulation fixed:

* `v0_baseline`  — the dialogue as in the main collection (internal reference).
* `v1_ack_alt`   — swaps the pre-filled ack for other forms ("Understood.",
  "Got it.", "Okay."). If the refusal is mimicry, the emitted *string* follows
  the ack; if it is behavior, the rate stays the same and the emitted form
  varies with it.
* `v2_no_ack`    — removes the assistant turns from the history and merges
  consecutive user turns. Keeps all the content, eliminates the example of a
  short assistant turn. If the refusal depends on seeing an assistant turn
  like that, it falls here.
* `v3_force`     — appends to the system prompt an explicit instruction that a
  letter is mandatory. Measures how much of the refusal survives being
  forbidden; whatever survives is the lower bound of the phenomenon.
"""
from __future__ import annotations

VARIANTS = ("v0_baseline", "v1_ack_alt", "v2_no_ack", "v3_force")

ACK_ALT = {"en": ["Understood.", "Got it.", "Okay."],
           "pt": ["Entendido.", "Certo.", "Ok."]}
FORCE = {
    "en": (" You must choose one of the four letters even if you disagree with "
           "the user or judge the question ambiguous. Acknowledging the message "
           "without giving a letter is not a valid answer."),
    "pt": (" Você deve escolher uma das quatro letras mesmo que discorde do "
           "usuário ou julgue a questão ambígua. Apenas acusar recebimento da "
           "mensagem, sem dar uma letra, não é resposta válida."),
}


def _merge_user_runs(msgs: list) -> list:
    out = []
    for m in msgs:
        if out and out[-1]["role"] == "user" and m["role"] == "user":
            out[-1] = {"role": "user",
                       "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(dict(m))
    return out


def apply_variant(msgs: list, variant: str, para: int = 0, lang: str = "en") -> list:
    """Return a new list of messages with the variant applied.

    Only the history block is touched: the probe turn (the last one) and the
    system turn remain the same, except in `v3_force`, which only touches the
    system turn.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if variant == "v0_baseline":
        return [dict(m) for m in msgs]
    if variant == "v3_force":
        out = [dict(m) for m in msgs]
        if out and out[0]["role"] == "system":
            out[0]["content"] = out[0]["content"] + FORCE[lang]
        else:                      # model without a system role: goes into the 1st user turn
            out[0] = dict(out[0])
            out[0]["content"] = out[0]["content"] + FORCE[lang]
        return out
    # the two variants that touch the history preserve the last turn (probe)
    head, probe = msgs[:-1], dict(msgs[-1])
    if variant == "v1_ack_alt":
        alt = ACK_ALT[lang][int(para) % 3]
        head = [dict(m, content=alt) if m["role"] == "assistant" else dict(m)
                for m in head]
        return head + [probe]
    if variant == "v2_no_ack":
        # preserves the assistant turn from the `doubt` condition (which carries the
        # model's baseline answer, not an ack)
        kept = [m for m in head
                if m["role"] != "assistant" or m["content"].strip().startswith(
                    ("Answer:", "Resposta:"))]
        # the probe enters the merge: the Mistral and Gemma-2 templates require
        # strict user/assistant alternation and reject two consecutive user
        # turns with TemplateError (the others tolerate it, which would produce
        # different structures across models for the same variant)
        return _merge_user_runs(kept + [probe])
    raise AssertionError
