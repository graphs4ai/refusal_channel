# When prefilled assistant turns produce non-answers: auditing multi-turn sycophancy scaffolds

Code, prompts and all generations for the paper of the same title (anonymous
submission). Every table, figure and in-text number of the paper is produced
by `scripts/20_paper_analysis.py` from the tables in `data/`.

## Layout

```
sycodef/          collection package (items, design grid, prompts, runner/cache,
                  vLLM backend, parser, prompt variants, metrics)
scripts/          numbered pipeline steps (see below)
config/           model configuration used for the runs (checkpoints, dtype, limits)
data/             released record tables (gzipped CSV) + item pools + run log
results/          CSV versions of every table (generated)
paper/            LaTeX source; paper/generated/ and paper/figures/ are generated
tests/            regression tests for the harness defects described in the paper
```

## Reproducing the paper from the released data (CPU only)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/20_paper_analysis.py      # results/, paper/generated/, paper/figures/
cd paper && latexmk -pdf sycophancy.tex
PYTHONPATH=. python -m pytest -q tests                 # optional
```

## Re-running the generations (GPU)

The runs used one 24 GB GPU (RTX 3090) with vLLM; see `requirements-gpu.txt`
and `data/generation_log.json` for versions, checkpoint revisions and timings.
Gated checkpoints (Llama-3.1, Gemma) need a Hugging Face token.

| step | script | output |
|---|---|---|
| items | `01_build_items.py` | probe and history item pools, calibration map |
| design | `02_build_design.py` | full design grid (`design.csv`) |
| throughput / shards | `03a`–`03d` | calibrated, resumable collection per model |
| merge | `06_progress_report.py --merge` | `responses.csv` (main records) |
| robustness arm | `11_run_variants.py` | prompt variants v0–v3 |
| auxiliary experiments I | `13_revision_experiments.py` | E1 matched no-pressure control, E2 self-generated history (v4), E3 refills and 256-token truncation re-runs |
| auxiliary experiments II | `14_review2_experiments.py` | v5 format-compatible system prompt, v6 no confidence field, v4b validated self-generated history, prompt-length audit (tokenizer only) |
| auxiliary experiments III | `15_review3_experiments.py` | v7 context-only system prompt (robustness grid), v8 credential wording control (full pressure grid, incorrect suggestions) |
| release tables | `19_assemble_release_data.py` | `data/*.csv.gz` |
| paper | `20_paper_analysis.py` | tables, figures, macros |
| parser check | `21_parser_check.py` | independent answer extraction by two open-weight models outside the panel (gpt-oss-20b and phi4-14b via Ollama): `build`, `annotate`, `collect` → `data/parser_check.csv`; the second checker was picked among four candidates on 60 outputs (`results/parser_check_candidates.csv`, `results/parser_check_candidates_input.jsonl`) |

Scripts `04`, `05`, `07`, `08`, `10` are earlier analyses (two-stage model,
power simulations) kept for transparency; the paper's numbers come from `20`.

Set `VLLM_USE_FLASHINFER_SAMPLER=0` if the FlashInfer JIT fails against an
older system `nvcc`.

## Released tables

| file | rows | content |
|---|---|---|
| `data/main_records.csv.gz` | 171,200 | pressure arm (128,000) and no-pressure arm (43,200): design factors, full response text, parsed option, parse rule, finish reason, timestamp |
| `data/robustness_records.csv.gz` | 115,200 | variants v0–v3, v5, v6, v7, and the two self-generated-history builds v4 and v4b (with the generated assistant turns) |
| `data/wording_control_records.csv.gz` | 32,000 | v8 credential wording control (200 probes × 4 histories × 5 prompt variants, incorrect suggestions) |
| `data/parser_check.csv` | 901 | options read by the two independent checkers for the parser-check sample (script 21) |
| `data/prompt_lengths.csv.gz` | 273,600 | prompt length in tokens after each model's chat template |
| `data/control_records.csv.gz` | 32,000 | E1 matched no-assertion control |
| `data/truncation_rerun.csv` | 133 | truncated non-answers, 64-token and 256-token texts |
| `data/items_*.csv`, `calibration_map.json` | | four-option items derived from MMLU-Pro and TruthfulQA |

Record identifiers encode item, cell, replicate, option order and paraphrase;
`scripts/20_paper_analysis.py` documents every definition used in the paper
(outcome categories, baseline, flip, contrasts, Holm families).

## Licences of the source data

Items are derived from MMLU-Pro (MIT) and TruthfulQA (Apache-2.0). Model
outputs are released for research on evaluation methodology. Code: MIT.
