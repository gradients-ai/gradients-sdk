from __future__ import annotations

import hashlib
import json
import logging
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from gradientsio.constants import DEFAULT_RUNPOD_GPU_COUNT
from gradientsio.constants import DEFAULT_RUNPOD_PORT
from gradientsio.constants import DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import DEFAULT_VLLM_MAX_MODEL_LEN
from gradientsio.constants import HUGGING_FACE_TOKEN_ENV
from gradientsio.constants import LARGE_MODEL_VLLM_MAX_MODEL_LEN
from gradientsio.constants import MEDIUM_MODEL_VLLM_MAX_MODEL_LEN
from gradientsio.errors import APIError
from gradientsio.errors import AuthenticationError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import NotFoundError
from gradientsio.errors import ValidationError
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentProvider
from gradientsio.models import LiumPod
from gradientsio.models import LiumTemplate
from gradientsio.models import RunPodPod


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _ResolvedModelRepos:
    base_model: str
    lora: str | None

    @property
    def served_model(self) -> str:
        return self.lora or self.base_model


def _resolve_model_repos(
    *,
    base_model: str,
    lora: str | None,
    hf_token: str | None,
) -> _ResolvedModelRepos:
    if not lora:
        return _ResolvedModelRepos(base_model=base_model, lora=None)
    if _adapter_base_model(lora, hf_token=hf_token):
        return _ResolvedModelRepos(base_model=base_model, lora=lora)
    return _ResolvedModelRepos(base_model=lora, lora=None)


def _adapter_base_model(repo_id: str, *, hf_token: str | None) -> str | None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]

        adapter_config_path = hf_hub_download(repo_id=repo_id, filename="adapter_config.json", token=hf_token)
    except Exception:
        return None

    try:
        with open(adapter_config_path) as config_file:
            adapter_config = json.load(config_file)
    except (OSError, ValueError):
        return None

    base_model = adapter_config.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def _build_vllm_env(
    *,
    base_model: str,
    lora: str | None,
    deployment_model_name: str,
    hf_token: str | None,
    env: dict[str, Any],
) -> dict[str, str]:
    result = {
        "MODEL_NAME": base_model,
    }
    if lora:
        lora_modules = [
            {
                "name": deployment_model_name,
                "path": lora,
                "base_model_name": base_model,
            }
        ]
        result["ENABLE_LORA"] = "true"
        result["LORA_MODULES"] = json.dumps(lora_modules, separators=(",", ":"))
    if hf_token:
        result[HUGGING_FACE_TOKEN_ENV] = hf_token
    for key, value in env.items():
        if value is not None:
            result[key] = str(value)
    return result


def _build_vllm_start_cmd(
    *,
    base_model: str,
    lora: str | None,
    deployment_model_name: str,
    port: int,
    host: str = "0.0.0.0",
    env: dict[str, Any] | None = None,
    hf_token: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | str | None = None,
    dtype: str | None = None,
    trust_remote_code: bool | None = None,
    enforce_eager: bool = False,
    max_lora_rank: int | None = DEFAULT_VLLM_MAX_LORA_RANK,
    gpu_count: int = DEFAULT_RUNPOD_GPU_COUNT,
) -> list[str]:
    env = env or {}
    resolved_max_model_len = str(
        max_model_len or _env_value(env, "MAX_MODEL_LEN") or _default_max_model_len(base_model, hf_token=hf_token)
    )
    resolved_gpu_memory_utilization = str(
        gpu_memory_utilization or _env_value(env, "GPU_MEMORY_UTILIZATION") or DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
    )
    resolved_dtype = dtype or _env_value(env, "DTYPE")
    resolved_trust_remote_code = (
        trust_remote_code if trust_remote_code is not None else _truthy_env_value(env, "TRUST_REMOTE_CODE")
    )
    command = [
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        base_model,
        "--served-model-name",
        deployment_model_name,
        "--max-model-len",
        resolved_max_model_len,
        "--gpu-memory-utilization",
        resolved_gpu_memory_utilization,
    ]
    if resolved_dtype:
        command.extend(["--dtype", resolved_dtype])
    if resolved_trust_remote_code:
        command.append("--trust-remote-code")
    if enforce_eager:
        command.append("--enforce-eager")
    if gpu_count > 1:
        command.extend(["--tensor-parallel-size", str(gpu_count)])
    if lora:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(max_lora_rank or DEFAULT_VLLM_MAX_LORA_RANK),
                "--lora-modules",
                f"{deployment_model_name}={lora}",
            ]
        )
    return command


def _default_gpu_count(model_repo: str) -> int:
    model_size_billions = _model_size_billions(model_repo)
    if model_size_billions is None:
        return DEFAULT_RUNPOD_GPU_COUNT
    if model_size_billions > 80:
        return 4
    if model_size_billions > 40:
        return 2
    return DEFAULT_RUNPOD_GPU_COUNT


def _default_max_model_len(model_repo: str, *, hf_token: str | None) -> int:
    model_size_billions = _model_size_billions(model_repo)
    if model_size_billions is not None and model_size_billions >= 30:
        return LARGE_MODEL_VLLM_MAX_MODEL_LEN
    if model_size_billions is not None and model_size_billions >= 14:
        return DEFAULT_VLLM_MAX_MODEL_LEN
    if model_size_billions is not None and model_size_billions >= 7:
        return MEDIUM_MODEL_VLLM_MAX_MODEL_LEN
    if model_size_billions is not None:
        return DEFAULT_VLLM_MAX_MODEL_LEN

    configured_context = _model_context_length(model_repo, hf_token=hf_token)
    if configured_context:
        return min(configured_context, DEFAULT_VLLM_MAX_MODEL_LEN)
    return DEFAULT_VLLM_MAX_MODEL_LEN


def _model_context_length(model_repo: str, *, hf_token: str | None) -> int | None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[reportMissingImports]

        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", token=hf_token)
    except Exception:
        return None

    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except (OSError, ValueError):
        return None

    for key in ("max_position_embeddings", "seq_length", "n_positions", "model_max_length"):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _model_size_billions(model_repo: str) -> float | None:
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*b(?:\D|$)", model_repo)
    if not match:
        return None
    return float(match.group(1))


def _env_value(env: dict[str, Any], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    return str(value)


def _truthy_env_value(env: dict[str, Any], key: str) -> bool:
    value = _env_value(env, key)
    return bool(value and value.lower() in {"1", "true", "yes", "on"})


def _deployment_key(
    *,
    provider: DeploymentProvider | str = DeploymentProvider.RUNPOD,
    base_model: str,
    lora: str | None,
    template_id: str,
    port: int,
    gpu_count: int,
    gpu_type_ids: list[str],
    env: dict[str, str],
    docker_start_cmd: list[str],
) -> str:
    public_env = {key: value for key, value in env.items() if key not in {HUGGING_FACE_TOKEN_ENV}}
    payload = {
        "provider": provider.value if isinstance(provider, DeploymentProvider) else str(provider),
        "base_model": base_model,
        "lora": lora,
        "template_id": template_id,
        "port": port,
        "gpu_count": gpu_count,
        "gpu_type_ids": gpu_type_ids,
        "env": public_env,
        "docker_start_cmd": docker_start_cmd,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _default_model_name(repo: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", repo).strip("-")
    return f"gradients-{slug}"[:63] or "gradients-model"


def _docker_image_name(image: str) -> str:
    return image.rsplit(":", 1)[0] if ":" in image.rsplit("/", 1)[-1] else image


def _docker_image_tag(image: str) -> str:
    return image.rsplit(":", 1)[1] if ":" in image.rsplit("/", 1)[-1] else "latest"


def _ensure_local_ssh_keypair() -> Path:
    ssh_dir = Path.home() / ".ssh"
    ed25519_public_key = ssh_dir / "id_ed25519.pub"
    rsa_public_key = ssh_dir / "id_rsa.pub"

    if ed25519_public_key.exists():
        logger.warning("Using local SSH public key for Lium deployment: %s", ed25519_public_key)
        return ed25519_public_key
    if rsa_public_key.exists():
        logger.warning("Using local SSH public key for Lium deployment: %s", rsa_public_key)
        return rsa_public_key

    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    private_key_path = ssh_dir / "id_ed25519"
    logger.warning("Creating SSH key pair for Lium deployment: %s", private_key_path)
    try:
        subprocess.run(
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-f",
                str(private_key_path),
                "-N",
                "",
                "-C",
                "gradients-sdk-lium",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(
            "Lium requires an SSH public key and the SDK could not create one with ssh-keygen. "
            "Create ~/.ssh/id_ed25519.pub manually and retry."
        ) from exc
    return ed25519_public_key


def _runpod_proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def _lium_server_url_from_pod(pod: LiumPod, internal_port: int) -> str:
    host = _lium_host_from_pod(pod)
    external_port = _lium_external_port(pod.ports_mapping, internal_port)
    if not host or external_port is None:
        return ""
    return f"http://{host}:{external_port}"


def _lium_host_from_pod(pod: LiumPod) -> str | None:
    if pod.ssh_connect_cmd:
        match = re.search(r"@(?P<host>[^\s]+)\s+-p\s+\d+", pod.ssh_connect_cmd)
        if match:
            return match.group("host")
    return None


def _lium_external_port(ports_mapping: dict[str, Any] | str | None, internal_port: int) -> int | None:
    if not ports_mapping:
        return None
    if isinstance(ports_mapping, str):
        try:
            ports_mapping = json.loads(ports_mapping)
        except ValueError:
            match = re.search(rf"{internal_port}[^0-9]+(?P<port>\d+)", ports_mapping)
            return int(match.group("port")) if match else None

    candidates = [
        str(internal_port),
        f"{internal_port}/tcp",
        f"{internal_port}/http",
    ]
    for key in candidates:
        if key in ports_mapping:
            port = _extract_external_port(ports_mapping[key])
            if port is not None:
                return port

    for key, value in ports_mapping.items():
        if str(internal_port) in str(key):
            port = _extract_external_port(value)
            if port is not None:
                return port
        if str(internal_port) in str(value):
            port = _extract_external_port(value)
            if port is not None:
                return port
    return None


def _extract_external_port(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value) if value.isdigit() else None
    if isinstance(value, list):
        for item in value:
            port = _extract_external_port(item)
            if port is not None:
                return port
    if isinstance(value, dict):
        for key in ("host_port", "HostPort", "external_port", "port", "published"):
            port_value = value.get(key)
            if isinstance(port_value, int):
                return port_value
            if isinstance(port_value, str) and port_value.isdigit():
                return int(port_value)
    return None


def _lium_template_env(template: LiumTemplate | dict[str, Any] | None) -> dict[str, str]:
    if template is None:
        return {}
    if isinstance(template, LiumTemplate):
        return template.environment or {}
    env = template.get("environment")
    return {str(key): str(value) for key, value in env.items()} if isinstance(env, dict) else {}


def _lium_internal_port_from_template(template: LiumTemplate | dict[str, Any] | None) -> int | None:
    ports: list[int] | None = None
    if isinstance(template, LiumTemplate):
        ports = template.internal_ports
    elif isinstance(template, dict) and isinstance(template.get("internal_ports"), list):
        ports = template["internal_ports"]
    if not ports:
        return None
    for port in ports:
        if port != 22:
            return int(port)
    return None


def _is_openai_endpoint_ready(server_url: str) -> bool:
    try:
        response = httpx.get(f"{server_url.rstrip('/')}/v1/models", timeout=10.0)
        if response.status_code != 200:
            return False
        payload = response.json()
        return isinstance(payload, dict) and isinstance(payload.get("data"), list)
    except (httpx.HTTPError, ValueError):
        return False


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


class _Spinner:
    frames = "|/-\\"

    def __init__(self, message: str) -> None:
        self.message = message
        self.index = 0
        self.enabled = sys.stderr.isatty()

    def sleep(self, seconds: float) -> None:
        if not self.enabled:
            time.sleep(seconds)
            return

        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._render()
            time.sleep(min(0.25, remaining))

    def finish(self, message: str | None = None) -> None:
        if not self.enabled:
            if message:
                logger.warning(message)
            return
        sys.stderr.write("\r" + " " * (len(self.message) + 8) + "\r")
        if message:
            sys.stderr.write(f"{message}\n")
        sys.stderr.flush()

    def _render(self) -> None:
        frame = self.frames[self.index % len(self.frames)]
        self.index += 1
        sys.stderr.write(f"\r{frame} {self.message}...")
        sys.stderr.flush()


def _log_deployment_url(details: DeploymentDetails) -> None:
    provider = details.provider.value if isinstance(details.provider, DeploymentProvider) else str(details.provider)
    logger.warning("%s deployment server URL: %s", provider.title(), details.server_url)


def _deployment_failure_message(details: DeploymentDetails) -> str | None:
    pod = details.pod
    if pod is None:
        return None
    signals = _collect_failure_signals(pod)
    if not signals:
        return None
    provider = details.provider.value if isinstance(details.provider, DeploymentProvider) else str(details.provider)
    return f"{provider.title()} deployment {details.id} failed while provisioning: {'; '.join(signals[:4])}"


def _collect_failure_signals(value: Any, *, path: str = "", seen: set[int] | None = None) -> list[str]:
    if value is None:
        return []
    seen = seen or set()
    if id(value) in seen:
        return []
    seen.add(id(value))

    if isinstance(value, dict):
        signals: list[str] = []
        for key, item in value.items():
            key_path = f"{path}.{key}" if path else str(key)
            signals.extend(_failure_signal_from_field(str(key), item, key_path))
            if isinstance(item, (dict, list, tuple)):
                signals.extend(_collect_failure_signals(item, path=key_path, seen=seen))
        return _dedupe_signals(signals)

    if isinstance(value, (list, tuple)):
        signals = []
        for index, item in enumerate(value):
            signals.extend(_collect_failure_signals(item, path=f"{path}[{index}]", seen=seen))
        return _dedupe_signals(signals)

    if hasattr(value, "model_dump"):
        try:
            return _collect_failure_signals(value.model_dump(), path=path, seen=seen)
        except Exception:
            return []
    return []


def _failure_signal_from_field(key: str, value: Any, path: str) -> list[str]:
    key_lower = key.lower()
    if isinstance(value, str):
        value_lower = value.lower()
        failure_terms = (
            "crashloop",
            "back-off",
            "backoff",
            "failed",
            "failure",
            "error",
            "oom",
            "out of memory",
            "exit code",
            "exited",
            "terminated",
        )
        if any(term in value_lower for term in failure_terms):
            return [f"{path}={value}"]
    if isinstance(value, int) and key_lower in {"exit_code", "exitcode", "exit_code"} and value != 0:
        return [f"{path}={value}"]
    return []


def _dedupe_signals(signals: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for signal in signals:
        if signal not in seen:
            deduped.append(signal)
            seen.add(signal)
    return deduped


def _port_from_pod(pod: RunPodPod) -> int:
    if pod.ports:
        first_port = pod.ports[0].split("/", 1)[0]
        if first_port.isdigit():
            return int(first_port)
    return DEFAULT_RUNPOD_PORT


def _raise_runpod_error(response: httpx.Response, payload: Any) -> None:
    _raise_provider_error("RunPod", response, payload)


def _raise_provider_error(provider: str, response: httpx.Response, payload: Any) -> None:
    message = response.reason_phrase
    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("message") or payload.get("detail")
        if detail is not None:
            message = str(detail)
        extra = payload.get("errors") or payload.get("details") or payload.get("validation_errors")
        if extra:
            try:
                extra_message = json.dumps(extra, sort_keys=True)
            except TypeError:
                extra_message = str(extra)
            message = f"{message}: {extra_message}"
        elif message.lower() in {"http error", "bad request", "unprocessable entity"}:
            try:
                payload_message = json.dumps(payload, sort_keys=True)
            except TypeError:
                payload_message = str(payload)
            message = f"{message}: {payload_message}"
    elif payload:
        message = str(payload)
    elif response.text:
        message = f"{message}: {response.text}"

    message = f"{provider} {response.status_code}: {message}"

    kwargs = {"status_code": response.status_code, "response": response}
    if response.status_code == 401:
        raise AuthenticationError(message, **kwargs)
    if response.status_code == 404:
        raise NotFoundError(message, **kwargs)
    if response.status_code in {400, 422}:
        raise ValidationError(message, **kwargs)
    raise APIError(f"{provider}: {message}", **kwargs)
