"""Local backends for the RTX 3090 (24 GB). Lazy import: the module loads
without vLLM or torch installed, so the analysis pipeline runs on the laptop.

What fits in 24 GB, with prompts of ~500 tokens and output of 64
(`max_model_len=1024` is generous; weights = params × bytes_per_parameter):

| model                     | precision| weights  | note                                |
|---------------------------|----------|----------|--------------------------------------|
| Mistral-7B-Instruct-v0.3  | bf16     | ~14.5 GB | plenty of room                       |
| Qwen2.5-7B-Instruct       | bf16     | ~15.2 GB | plenty of room                       |
| Llama-3.1-8B-Instruct     | bf16     | ~16.1 GB | ok, `gpu_memory_utilization=0.92`    |
| Gemma-2-9B-it             | bf16     | ~18.5 GB | tight; no system role in the template|
| Qwen2.5-14B-Instruct-AWQ  | 4-bit    | ~9.5 GB  | needs quantization                   |
| Qwen2.5-32B-Instruct-AWQ  | 4-bit    | ~19.5 GB | fits, but is the slowest shard       |

Two pitfalls worth the care:

1. **Prefix cache.** Records from the same (item, history, paraphrase)
   share 12 calibration messages. With `enable_prefix_caching=True` and
   jobs sorted by `prefix_key`, the prefill of these messages is paid once
   per group instead of once per record — it's the single biggest saving
   in the collection.
2. **Template without a `system` role.** Gemma-2 (among others) rejects
   messages with `role="system"`. `normalize_messages` folds the system
   message into the first user turn when `supports_system=False`.
"""
from __future__ import annotations

from .runner import Backend

# Models whose chat template does not accept the `system` role.
NO_SYSTEM_ROLE = ("gemma", "gemma-2", "gemma-3")


def supports_system_role(model_name: str) -> bool:
    return not any(k in model_name.lower() for k in NO_SYSTEM_ROLE)


def normalize_messages(messages: list, supports_system: bool = True) -> list:
    """Folds the system turn into the first user turn when needed."""
    if supports_system or not messages or messages[0]["role"] != "system":
        return messages
    sys_txt = messages[0]["content"]
    rest = messages[1:]
    if not rest:
        return [{"role": "user", "content": sys_txt}]
    head = dict(rest[0])
    head["content"] = sys_txt + "\n\n" + head["content"]
    return [head] + rest[1:]


class VLLMBackend(Backend):
    """Offline batch inference. One instance per model, loaded once
    per shard run (the loading cost goes into `Throughput.load_s`).
    """
    name = "vllm"

    def __init__(self, model: str, dtype: str = "bfloat16",
                 quantization: str | None = None, max_model_len: int = 1024,
                 gpu_memory_utilization: float = 0.90, max_num_seqs: int = 64,
                 enable_prefix_caching: bool = True, seed: int = 0,
                 trust_remote_code: bool = False,
                 chat_template_kwargs: dict | None = None):
        from vllm import LLM                       # lazy import
        from transformers import AutoTokenizer
        self.model_name = model
        self.supports_system = supports_system_role(model)
        # Qwen3 enters reasoning mode by default and would spend the 64 output
        # tokens on the <think> block; `{"enable_thinking": False}` turns this
        # off in the template itself, keeping the output comparable to the
        # other models.
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tok = AutoTokenizer.from_pretrained(model,
                                                 trust_remote_code=trust_remote_code)
        self.llm = LLM(model=model, dtype=dtype, quantization=quantization,
                       max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization,
                       max_num_seqs=max_num_seqs,
                       enable_prefix_caching=enable_prefix_caching, seed=seed,
                       trust_remote_code=trust_remote_code)

    def _render(self, messages: list) -> str:
        msgs = normalize_messages(messages, self.supports_system)
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True,
                                            **self.chat_template_kwargs)

    def generate_batch(self, messages_list: list, temperature: float = 0.0,
                       max_tokens: int = 64, seeds: list | None = None,
                       stop: list | None = None, min_tokens: int = 0, **_):
        """`stop` (strings) and `min_tokens` are only used by the self-generated
        history turns of script 14; the main collection used neither."""
        from vllm import SamplingParams
        prompts = [self._render(m) for m in messages_list]
        sps = [SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              seed=(seeds[i] if seeds else None),
                              stop=stop or None, min_tokens=min_tokens,
                              include_stop_str_in_output=False)
               for i in range(len(prompts))]
        outs = self.llm.generate(prompts, sps)
        return [{"text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
                 "logprobs": None} for o in outs]

    def generate(self, model, messages, **params):
        return self.generate_batch([messages], **params)[0]

    def max_prompt_tokens(self, messages_list: list) -> int:
        return max(len(self.tok(self._render(m))["input_ids"]) for m in messages_list)


class HFBackend(Backend):
    """Fallback using `transformers`. Slower than vLLM (no paged attention or
    prefix cache), but useful for a model that vLLM doesn't support yet."""
    name = "hf"

    def __init__(self, model: str, dtype: str = "bfloat16", device: str = "cuda",
                 load_in_4bit: bool = False, trust_remote_code: bool = False,
                 chat_template_kwargs: dict | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model
        self.supports_system = supports_system_role(model)
        # Qwen3 enters reasoning mode by default and would spend the 64 output
        # tokens on the <think> block; `{"enable_thinking": False}` turns this
        # off in the template itself, keeping the output comparable to the
        # other models.
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left",
                                                 trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = {"torch_dtype": getattr(torch, dtype), "device_map": device,
              "trust_remote_code": trust_remote_code}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kw.pop("torch_dtype")
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        self.torch = torch

    def generate_batch(self, messages_list: list, temperature: float = 0.0,
                       max_tokens: int = 64, seeds: list | None = None, **_):
        prompts = [self.tok.apply_chat_template(
            normalize_messages(m, self.supports_system), tokenize=False,
            add_generation_prompt=True) for m in messages_list]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        if seeds and seeds[0] is not None:
            self.torch.manual_seed(int(seeds[0]))
        with self.torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_tokens,
                do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        return [{"text": t, "finish": "stop", "logprobs": None} for t in texts]

    def generate(self, model, messages, **params):
        return self.generate_batch([messages], **params)[0]


class BatchedStub(Backend):
    """Synthetic model with a batch interface, to exercise the shard
    path without a GPU."""
    name = "batched_stub"

    def __init__(self, **cfg):
        from .runner import StubBackend
        self.inner = StubBackend(**cfg)

    def generate_batch(self, messages_list, temperature=0.0, max_tokens=64,
                       seeds=None, metas=None, **_):
        metas = metas or [None] * len(messages_list)
        return [self.inner.generate("stub", m, meta=meta)
                for m, meta in zip(messages_list, metas)]

    def generate(self, model, messages, meta=None, **params):
        return self.inner.generate(model, messages, meta=meta)
