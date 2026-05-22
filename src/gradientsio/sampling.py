from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from gradientsio.constants import HUGGING_FACE_HUB_TOKEN_ENV
from gradientsio.constants import HUGGING_FACE_TOKEN_ENV
from gradientsio.models import DeploymentProvider


TRANSFORMERS_BACKEND = "transformers"
DEFAULT_TOP_P = 1.0
DEFAULT_N = 1
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_FREQUENCY_PENALTY = 0.0


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
        backend: DeploymentProvider | str = DeploymentProvider.LOCAL,
        hf_token: str | None = None,
        require_cuda: bool = True,
        trust_remote_code: bool = True,
        server_url: str | None = None,
        server_model: str | None = None,
        keep_server_alive: bool = False,
        vllm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.backend = _coerce_sampler_backend(backend)
        self.hf_token = hf_token or os.getenv(HUGGING_FACE_TOKEN_ENV) or os.getenv(HUGGING_FACE_HUB_TOKEN_ENV)
        self.require_cuda = require_cuda
        self.trust_remote_code = trust_remote_code
        self.server_url = server_url
        self.server_model = server_model
        self.keep_server_alive = keep_server_alive
        self.vllm_kwargs = vllm_kwargs or {}
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

    def generate(
        self,
        model_repo: str,
        prompts: list[str],
        *,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if self.backend == DeploymentProvider.LOCAL:
            return self._generate_with_vllm(
                base_model_repo=model_repo,
                lora_repo=None,
                prompts=prompts,
                config=config,
                **kwargs,
            )
        if kwargs:
            raise ValueError("OpenAI/vLLM request parameters are only supported for the local vLLM backend.")
        return self._timed_generate("model", model_repo, prompts, config=config)

    def generate_with_adapter(
        self,
        adapter_or_model_repo: str,
        prompts: list[str],
        *,
        base_model_repo: str,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if self.backend == DeploymentProvider.LOCAL:
            return self._generate_with_vllm(
                base_model_repo=base_model_repo,
                lora_repo=adapter_or_model_repo,
                prompts=prompts,
                config=config,
                **kwargs,
            )
        if kwargs:
            raise ValueError("OpenAI/vLLM request parameters are only supported for the local vLLM backend.")
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

    def _generate_with_vllm(
        self,
        *,
        base_model_repo: str,
        lora_repo: str | None,
        prompts: list[str],
        config: GenerationConfig | None,
        **kwargs: Any,
    ) -> list[str]:
        if self.server_url:
            model = self.server_model or lora_repo or base_model_repo
            return RemoteVLLMSampler(base_url=self.server_url, model=model).generate(prompts, config=config, **kwargs)

        from gradientsio.deployment import deploy_local_vllm

        deployment = deploy_local_vllm(
            base_model=base_model_repo,
            lora=lora_repo,
            hf_token=self.hf_token,
            trust_remote_code=self.trust_remote_code,
            **self.vllm_kwargs,
        )
        try:
            return deployment.sampler().generate(prompts, config=config, **kwargs)
        finally:
            if deployment.started_by_sdk and not self.keep_server_alive:
                deployment.stop()


def _torch() -> Any:
    import torch  # type: ignore[reportMissingImports]

    return torch


def _coerce_sampler_backend(backend: DeploymentProvider | str) -> DeploymentProvider | str:
    if isinstance(backend, DeploymentProvider):
        return backend
    normalized = backend.lower()
    if normalized == "vllm":
        return DeploymentProvider.LOCAL
    if normalized == TRANSFORMERS_BACKEND:
        return TRANSFORMERS_BACKEND
    return DeploymentProvider(normalized)


def _set_optional_payload_fields(payload: dict[str, Any], **fields: Any) -> None:
    for key, value in fields.items():
        if value is not None:
            payload[key] = value


class RemoteVLLMSampler:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = 60.0,
        verify_ssl: bool = True,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(timeout=timeout, verify=verify_ssl)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def generate(
        self,
        prompts: list[str],
        *,
        config: GenerationConfig | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float = DEFAULT_TOP_P,
        n: int = DEFAULT_N,
        stream: bool = False,
        stop: str | list[str] | None = None,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
        repetition_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        user: str | None = None,
        seed: int | None = None,
        best_of: int | None = None,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        echo: bool = False,
        suffix: str | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        min_tokens: int | None = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        if stream:
            raise NotImplementedError("Streaming responses are not supported by RemoteVLLMSampler.generate().")
        config = config or GenerationConfig()
        resolved_temperature = temperature
        if resolved_temperature is None:
            resolved_temperature = 0 if not config.do_sample else 0.7
        payload = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": max_tokens or config.max_new_tokens,
            "temperature": resolved_temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repetition_penalty": repetition_penalty or config.repetition_penalty,
            "echo": echo,
            "ignore_eos": ignore_eos,
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
            **kwargs,
        }
        _set_optional_payload_fields(
            payload,
            stop=stop,
            logit_bias=logit_bias,
            user=user,
            seed=seed,
            best_of=best_of,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            suffix=suffix,
            top_k=top_k,
            min_p=min_p,
            min_tokens=min_tokens,
        )
        response = self._post("/v1/completions", payload)
        choices = response.get("choices", [])
        return [str(choice.get("text", "")) for choice in choices]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        config: GenerationConfig | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float = DEFAULT_TOP_P,
        n: int = DEFAULT_N,
        stream: bool = False,
        stop: str | list[str] | None = None,
        presence_penalty: float = DEFAULT_PRESENCE_PENALTY,
        frequency_penalty: float = DEFAULT_FREQUENCY_PENALTY,
        repetition_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        user: str | None = None,
        seed: int | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        min_tokens: int | None = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        if stream:
            raise NotImplementedError("Streaming responses are not supported by RemoteVLLMSampler.chat().")
        config = config or GenerationConfig()
        resolved_temperature = temperature
        if resolved_temperature is None:
            resolved_temperature = 0 if not config.do_sample else 0.7
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or config.max_new_tokens,
            "temperature": resolved_temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "repetition_penalty": repetition_penalty or config.repetition_penalty,
            "ignore_eos": ignore_eos,
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
            **kwargs,
        }
        _set_optional_payload_fields(
            payload,
            stop=stop,
            logit_bias=logit_bias,
            user=user,
            seed=seed,
            top_k=top_k,
            min_p=min_p,
            min_tokens=min_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
        response = self._post("/v1/chat/completions", payload)
        choices = response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content", ""))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(f"{self.base_url}/{path.lstrip('/')}", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
