"""sycodef — single collection pass for P1 (deference weighted by demonstrated
reliability) and P2 (measured null for sycophancy). COLING 2027 / ARR 12-10-2026."""
from . import (analysis, bayes, collect, design, extract, items, metrics,  # noqa: F401
               prompts, runner, shards, simulate)

# backends_gpu is excluded from automatic import: depends on vllm/torch
__all__ = ["items", "design", "prompts", "collect", "runner", "extract",
           "bayes", "metrics", "analysis", "simulate", "shards"]
__version__ = "0.1.0"
