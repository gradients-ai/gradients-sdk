from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 96
    do_sample: bool = False
    repetition_penalty: float = 1.12
    num_beams: int = 4
    max_input_tokens: int = 3072


def load_dataset_rows(
    dataset: str,
    *,
    split: str = "train",
    sample_size: int = 5,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset  # type: ignore[reportMissingImports]

    rows = load_dataset(dataset, split=split)
    if seed is not None:
        rows = rows.shuffle(seed=seed)
    if sample_size:
        rows = rows.select(range(sample_size))
    return [dict(row) for row in rows]


class ModelSampler:
    def __init__(
        self,
        *,
        hf_token: str | None = None,
        require_cuda: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self.require_cuda = require_cuda
        self.trust_remote_code = trust_remote_code
        from huggingface_hub import HfApi  # type: ignore[reportMissingImports]

        self._api = HfApi(token=self.hf_token)

    def cuda_status(self) -> str:
        torch = _torch()
        if torch.cuda.is_available():
            return f"CUDA OK - {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}"
        if self.require_cuda:
            raise RuntimeError(
                "CUDA is required, but PyTorch does not see a GPU. Install a CUDA-enabled PyTorch build "
                "and confirm `nvidia-smi` works in the same environment."
            )
        return "Warning: CUDA unavailable; inference will use CPU."

    def generate(self, model_repo: str, prompts: list[str], *, config: GenerationConfig | None = None) -> list[str]:
        return self._timed_generate("model", model_repo, prompts, config=config)

    def generate_with_adapter(
        self,
        adapter_or_model_repo: str,
        prompts: list[str],
        *,
        base_model_repo: str,
        config: GenerationConfig | None = None,
    ) -> list[str]:
        return self._timed_generate(
            "adapter",
            adapter_or_model_repo,
            prompts,
            base_model_repo=base_model_repo,
            config=config,
        )

    def load_model(self, repo_id: str, *, base_model_repo: str | None = None) -> tuple[Any, Any]:
        from peft import PeftModel  # type: ignore[reportMissingImports]
        from transformers import AutoModelForCausalLM  # type: ignore[reportMissingImports]

        self.cuda_status()
        if base_model_repo and self.repo_has_file(repo_id, "adapter_config.json"):
            print(f"Loading base model {base_model_repo} and merging adapter {repo_id}...")
            tokenizer = self.load_tokenizer(base_model_repo)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_repo, **self.model_kwargs())
            model = PeftModel.from_pretrained(base_model, repo_id, token=self.hf_token)
            model = model.merge_and_unload()
            model.eval()
            return tokenizer, model

        print(f"Loading full model {repo_id}...")
        tokenizer = self.load_tokenizer(repo_id)
        model = AutoModelForCausalLM.from_pretrained(repo_id, **self.model_kwargs())
        model.eval()
        return tokenizer, model

    def generate_with_model(
        self,
        tokenizer: Any,
        model: Any,
        prompts: list[str],
        *,
        config: GenerationConfig | None = None,
    ) -> list[str]:
        torch = _torch()
        config = config or GenerationConfig()
        answers = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_input_tokens)
            device = next(model.parameters()).device
            inputs = {name: value.to(device) for name, value in inputs.items()}

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": config.max_new_tokens,
                "do_sample": config.do_sample,
                "pad_token_id": tokenizer.eos_token_id,
                "repetition_penalty": config.repetition_penalty,
            }
            if config.num_beams > 1:
                gen_kwargs["num_beams"] = config.num_beams
                gen_kwargs["early_stopping"] = True

            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs)

            new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
            answers.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return answers

    def repo_has_file(self, repo_id: str, filename: str) -> bool:
        try:
            info = self._api.model_info(repo_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read model repo {repo_id!r}. If it is private, set HF_TOKEN before running inference."
            ) from exc
        return any(sibling.rfilename == filename for sibling in info.siblings)

    def load_tokenizer(self, repo_id: str) -> Any:
        from transformers import AutoTokenizer  # type: ignore[reportMissingImports]

        kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def model_kwargs(self) -> dict[str, Any]:
        torch = _torch()
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["torch_dtype"] = torch.bfloat16
        else:
            kwargs["torch_dtype"] = torch.float32
        return kwargs

    @staticmethod
    def release_model(model: Any | None = None) -> None:
        torch = _torch()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _timed_generate(
        self,
        label: str,
        repo: str,
        prompts: list[str],
        *,
        config: GenerationConfig | None,
        base_model_repo: str | None = None,
    ) -> list[str]:
        print(f"[{label}] Loading {repo!r} ({len(prompts)} prompts)...", flush=True)
        started = time.perf_counter()
        tokenizer, model = self.load_model(repo, base_model_repo=base_model_repo)
        try:
            answers = self.generate_with_model(tokenizer, model, prompts, config=config)
        finally:
            self.release_model(model)
        print(f"[{label}] Finished in {time.perf_counter() - started:.1f}s", flush=True)
        return answers


def _torch() -> Any:
    import torch  # type: ignore[reportMissingImports]

    return torch
