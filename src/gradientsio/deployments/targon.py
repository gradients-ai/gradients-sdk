from __future__ import annotations

import shlex
import time
from typing import TYPE_CHECKING
from typing import Any

import httpx

from gradientsio.constants import DEFAULT_TARGON_PORT
from gradientsio.constants import DEFAULT_TARGON_PROJECT_NAME
from gradientsio.constants import DEFAULT_TARGON_RESOURCE_PREFERENCES
from gradientsio.constants import DEFAULT_VLLM_MAX_LORA_RANK
from gradientsio.constants import RUNPOD_VLLM_IMAGE_NAME
from gradientsio.constants import TARGON_API_KEY_ENV
from gradientsio.deployments.common import _build_vllm_env
from gradientsio.deployments.common import _build_vllm_start_cmd
from gradientsio.deployments.common import _default_gpu_count
from gradientsio.deployments.common import _default_model_name
from gradientsio.deployments.common import _deployment_key
from gradientsio.deployments.common import _log_deployment_url
from gradientsio.deployments.common import _raise_provider_error
from gradientsio.deployments.common import _resolve_model_repos
from gradientsio.deployments.common import _Spinner
from gradientsio.deployments.common import logger
from gradientsio.errors import APIError
from gradientsio.errors import ConfigurationError
from gradientsio.errors import NetworkError
from gradientsio.errors import NotFoundError
from gradientsio.errors import TaskTimeout
from gradientsio.models import DeploymentDetails
from gradientsio.models import DeploymentProvider
from gradientsio.models import DeploymentStatus
from gradientsio.models import TargonApp
from gradientsio.sampling import RemoteVLLMSampler


if TYPE_CHECKING:
    from gradientsio.deployments.client import DeploymentClient


class TargonDeployment:
    def __init__(
        self,
        deployments: DeploymentClient,
        details: DeploymentDetails,
        *,
        model_name: str,
        inference_api_key: str | None = None,
    ) -> None:
        self._deployments = deployments
        self.details = details
        self.model_name = model_name
        self.inference_api_key = inference_api_key

    @property
    def id(self) -> str:
        return self.details.id

    @property
    def deployment_key(self) -> str:
        return self.details.deployment_key

    @property
    def server_url(self) -> str:
        return self.details.server_url

    def refresh(self) -> DeploymentDetails:
        self.details = self._deployments.get_targon(self.id, deployment_key=self.deployment_key)
        return self.details

    def status(self) -> DeploymentStatus | str:
        return self.refresh().status

    def delete(self) -> None:
        self._deployments.delete_targon(self.id)

    def sampler(self, *, timeout: float = 60.0, verify_ssl: bool = True) -> RemoteVLLMSampler:
        return RemoteVLLMSampler(
            base_url=self.server_url,
            model=self.model_name,
            api_key=self.inference_api_key,
            timeout=timeout,
            verify_ssl=verify_ssl,
        )

    def wait_ready(
        self,
        *,
        timeout: float | None = None,
        poll_interval: float = 15.0,
    ) -> DeploymentDetails:
        started_at = time.monotonic()
        logger.warning("Deploying on Targon now. This may take a few minutes, please grab a coffee.")
        spinner = _Spinner("Waiting for Targon vLLM server to become ready")

        while True:
            details = self.refresh()
            if details.is_running and self._deployments.is_targon_endpoint_ready(self.server_url):
                spinner.finish("Targon deployment is ready.")
                return details

            if details.is_terminal:
                spinner.finish()
                raise APIError(f"Targon deployment {self.id} reached terminal status {details.status}")
            failure_message = self._deployments.targon_failure_reason(self.id)
            if failure_message:
                spinner.finish()
                raise APIError(failure_message)

            if timeout is not None and time.monotonic() - started_at >= timeout:
                spinner.finish()
                raise TaskTimeout(f"Targon deployment {self.id} was not ready within {timeout} seconds")

            sleep_for = poll_interval
            if timeout is not None:
                remaining = timeout - (time.monotonic() - started_at)
                sleep_for = max(0.0, min(poll_interval, remaining))
            spinner.sleep(sleep_for)


class TargonDeploymentMixin:
    def deploy_targon(
        self,
        *,
        base_model: str,
        lora: str | None = None,
        deployment_model_name: str | None = None,
        hf_token: str | None = None,
        env: dict[str, Any] | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool | None = None,
        enforce_eager: bool = False,
        max_lora_rank: int | None = DEFAULT_VLLM_MAX_LORA_RANK,
        gpu_count: int | None = None,
        port: int = DEFAULT_TARGON_PORT,
        resource: str | list[str] | tuple[str, ...] | None = None,
        project_name: str = DEFAULT_TARGON_PROJECT_NAME,
        app_name: str | None = None,
        min_replicas: int = 1,
        max_replicas: int = 1,
        timeout: int = 1800,
        startup_timeout: int | None = 1800,
        requires_auth: bool = False,
        wait: bool = True,
        reuse_existing: bool = True,
        inference_api_key: str | None = None,
    ) -> TargonDeployment:
        self._require_targon_api_key()
        resource_preferences = _targon_resource_preferences(resource)
        resolved = _resolve_model_repos(base_model=base_model, lora=lora, hf_token=hf_token)
        resolved_gpu_count = gpu_count or _default_gpu_count(resolved.base_model)
        deployment_model_name = deployment_model_name or _default_model_name(resolved.served_model)
        vllm_env = _build_vllm_env(
            base_model=resolved.base_model,
            lora=resolved.lora,
            deployment_model_name=deployment_model_name,
            hf_token=hf_token,
            env=env or {},
        )
        vllm_args = _build_vllm_start_cmd(
            base_model=resolved.base_model,
            lora=resolved.lora,
            deployment_model_name=deployment_model_name,
            port=port,
            env=env or {},
            hf_token=hf_token,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
            max_lora_rank=max_lora_rank,
            gpu_count=resolved_gpu_count,
        )
        targon_vllm_args = _build_targon_vllm_args(resolved.base_model, vllm_args)
        targon_container_args = _targon_runtime_port_args(targon_vllm_args, fallback_port=port)
        selected_resource = self._select_targon_resource(resource_preferences)
        deployment_key = _deployment_key(
            provider=DeploymentProvider.TARGON,
            base_model=resolved.base_model,
            lora=resolved.lora,
            template_id=RUNPOD_VLLM_IMAGE_NAME,
            port=port,
            gpu_count=resolved_gpu_count,
            gpu_type_ids=[selected_resource],
            env=vllm_env,
            docker_start_cmd=targon_container_args,
        )
        resolved_app_name = app_name or _default_targon_app_name(deployment_key)
        if len(resolved_app_name) > 32:
            raise ConfigurationError("Targon app_name must be at most 32 characters long.")

        if reuse_existing:
            existing = self._find_targon_deployment(deployment_key, resolved_app_name)
            if existing:
                return self._targon_deployment_from_app(
                    existing,
                    deployment_key=deployment_key,
                    base_model=resolved.base_model,
                    lora=resolved.lora,
                    model_name=deployment_model_name,
                    inference_api_key=inference_api_key,
                )

        metadata_env = {
            "GRADIENTS_DEPLOYMENT_KEY": deployment_key,
            "GRADIENTS_BASE_MODEL": resolved.base_model,
            "GRADIENTS_SDK_PROVIDER": DeploymentProvider.TARGON.value,
            "GRADIENTS_TARGON_RESOURCE": selected_resource,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
        if resolved.lora:
            metadata_env["GRADIENTS_LORA"] = resolved.lora

        app = self._deploy_targon_container(
            app_name=resolved_app_name,
            resource=selected_resource,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            port=port,
            startup_timeout=startup_timeout,
            requires_auth=requires_auth,
            env={**vllm_env, **metadata_env},
            vllm_args=targon_container_args,
        )
        deployment = self._targon_deployment_from_app(
            app,
            deployment_key=deployment_key,
            base_model=resolved.base_model,
            lora=resolved.lora,
            model_name=deployment_model_name,
            inference_api_key=inference_api_key,
        )
        if wait:
            deployment.wait_ready()
        return deployment

    def get_targon(self, app_id: str, *, deployment_key: str | None = None) -> DeploymentDetails:
        app = _targon_app_from_workload(self._targon_request("GET", f"/tha/v2/workloads/{app_id}"))
        key = deployment_key or _deployment_key_from_targon_app(app)
        return DeploymentDetails(
            provider=DeploymentProvider.TARGON,
            id=app.uid,
            deployment_key=key,
            base_model=str(app.metadata.get("GRADIENTS_BASE_MODEL") or ""),
            lora=app.metadata.get("GRADIENTS_LORA"),
            status=app.normalized_status,
            server_url=_targon_server_url_from_app(app),
            pod=app,
        )

    def delete_targon(self, app_id: str) -> None:
        self._targon_request("DELETE", f"/tha/v2/workloads/{app_id}")

    def is_targon_endpoint_ready(self, server_url: str) -> bool:
        if not server_url:
            return False
        try:
            response = self._client.get(f"{server_url.rstrip('/')}/v1/models", timeout=10.0, follow_redirects=True)
            if response.status_code != 200:
                return False
            payload = response.json()
            return isinstance(payload, dict) and isinstance(payload.get("data"), list)
        except httpx.HTTPError:
            try:
                response = httpx.get(
                    f"{server_url.rstrip('/')}/v1/models",
                    timeout=10.0,
                    verify=False,
                    follow_redirects=True,
                )
                if response.status_code != 200:
                    return False
                payload = response.json()
                return isinstance(payload, dict) and isinstance(payload.get("data"), list)
            except (httpx.HTTPError, ValueError):
                return False
        except ValueError:
            return False

    def targon_failure_reason(self, app_id: str) -> str | None:
        try:
            state = self._targon_request("GET", f"/tha/v2/workloads/{app_id}/state")
            events = self._targon_request("GET", f"/tha/v2/workloads/{app_id}/events", params={"limit": 50})
            logs = self._targon_logs(app_id)
        except Exception as exc:
            logger.debug("Could not fetch Targon deployment diagnostics for %s: %s", app_id, exc)
            return None
        return _targon_failure_reason(app_id=app_id, state=state, events=events, logs=logs)

    def _find_targon_deployment(self, deployment_key: str, app_name: str) -> TargonApp | None:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 100, "type": "SERVERLESS"}
            if cursor:
                params["cursor"] = cursor
            payload = self._targon_request("GET", "/tha/v2/workloads", params=params)
            items = payload.get("items", payload) if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                return None
            for item in items:
                app = _targon_app_from_workload(item)
                if app.name == app_name or _deployment_key_from_targon_app(app) == deployment_key:
                    server_url = _targon_server_url_from_app(app)
                    if server_url:
                        app.web_url = server_url
                        return app
                    logger.warning("Ignoring incomplete Targon workload without a web endpoint: %s", app.uid)
            cursor = str(payload.get("next_cursor") or "") if isinstance(payload, dict) else ""
            if not cursor:
                return None

    def _select_targon_resource(self, resource_preferences: list[str]) -> str:
        try:
            return _select_targon_resource_from_inventory(
                request=self._targon_request,
                resource_preferences=resource_preferences,
            )
        except NotFoundError:
            raise
        except Exception as exc:
            resources = ", ".join(resource_preferences)
            raise NotFoundError(f"No available Targon GPU resource found for preferences: {resources}") from exc

    def _targon_deployment_from_app(
        self,
        app: TargonApp,
        *,
        deployment_key: str,
        base_model: str,
        lora: str | None,
        model_name: str,
        inference_api_key: str | None,
    ) -> TargonDeployment:
        details = DeploymentDetails(
            provider=DeploymentProvider.TARGON,
            id=app.uid,
            deployment_key=deployment_key,
            base_model=base_model,
            lora=lora,
            status=app.normalized_status,
            server_url=_targon_server_url_from_app(app),
            pod=app,
        )
        _log_deployment_url(details)
        return TargonDeployment(self, details, model_name=model_name, inference_api_key=inference_api_key)

    def _deploy_targon_container(
        self,
        *,
        app_name: str,
        resource: str,
        min_replicas: int,
        max_replicas: int,
        port: int,
        startup_timeout: int | None,
        requires_auth: bool,
        env: dict[str, str],
        vllm_args: list[str],
    ) -> TargonApp:
        response = _deploy_targon_serverless_container(
            request=self._targon_request,
            app_name=app_name,
            resource=resource,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            port=port,
            startup_timeout=startup_timeout,
            requires_auth=requires_auth,
            env=env,
            vllm_args=vllm_args,
        )
        app = _targon_app_from_workload(response)
        if not app.metadata:
            app.metadata = env
        return app

    def _targon_request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        self._require_targon_api_key()
        url = f"{self.targon_base_url}/{path.lstrip('/')}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.targon_api_key}",
        }
        if json is not None:
            headers["Content-Type"] = "application/json"

        try:
            response = self._client.request(method, url, headers=headers, json=json, params=params)
        except httpx.HTTPError as exc:
            raise NetworkError(str(exc)) from exc

        if response.status_code >= 400:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = {}
            _raise_provider_error("Targon", response, error_payload)

        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise NetworkError(f"Targon returned a non-JSON response from {path}") from exc

    def _targon_logs(self, app_id: str) -> str:
        self._require_targon_api_key()
        url = f"{self.targon_base_url}/tha/v2/workloads/{app_id}/logs"
        headers = {
            "Accept": "text/plain",
            "Authorization": f"Bearer {self.targon_api_key}",
        }
        try:
            response = self._client.request("GET", url, headers=headers, params={"follow": "false"})
        except httpx.HTTPError:
            return ""
        if response.status_code >= 400:
            return ""
        return response.text

    def _require_targon_api_key(self) -> None:
        if not self.targon_api_key:
            raise ConfigurationError(f"Set {TARGON_API_KEY_ENV} in your environment to deploy on Targon.")


def _build_targon_vllm_args(base_model: str, vllm_args: list[str]) -> list[str]:
    command = [base_model, "--uvicorn-log-level=info"]
    index = 0
    while index < len(vllm_args):
        arg = vllm_args[index]
        if arg == "--model":
            index += 2
            continue
        command.append(arg)
        index += 1
    return command


def _targon_runtime_port_args(vllm_args: list[str], *, fallback_port: int) -> list[str]:
    args = list(vllm_args)
    for index, arg in enumerate(args):
        if arg == "--port" and index + 1 < len(args):
            args[index + 1] = f"${{PORT:-{fallback_port}}}"
            return args
    args.extend(["--port", f"${{PORT:-{fallback_port}}}"])
    return args


def _shell_join_with_runtime_port(command: list[str]) -> str:
    return " ".join(
        f'"{part}"' if part.startswith("${PORT:-") and part.endswith("}") else shlex.quote(part)
        for part in command
    )


def _targon_resource_preferences(resource: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if resource is None:
        return list(DEFAULT_TARGON_RESOURCE_PREFERENCES)
    if isinstance(resource, str):
        return [resource]
    resources = [str(item) for item in resource if item]
    if not resources:
        raise ConfigurationError("At least one Targon resource must be provided.")
    return resources


def _default_targon_app_name(deployment_key: str) -> str:
    return f"grad-targon-{deployment_key}"


def _select_targon_resource_from_inventory(
    *,
    request: Any,
    resource_preferences: list[str],
) -> str:
    payload = request("GET", "/tha/v2/inventory", params={"type": "serverless", "gpu": "true"})
    inventory = payload.get("items", payload) if isinstance(payload, dict) else payload
    if not isinstance(inventory, list):
        raise APIError("Targon returned an unexpected inventory response.")

    available = {
        str(item.get("name")): item
        for item in inventory
        if isinstance(item, dict) and item.get("name") and int(item.get("available") or 0) > 0
    }
    for resource in resource_preferences:
        if resource in available:
            logger.warning(
                "Selected Targon resource %s (%s available)",
                resource,
                available[resource].get("available"),
            )
            return resource
    available_names = ", ".join(sorted(available)) or "none"
    raise NotFoundError(
        f"No preferred Targon resource is available. Available GPU resources: {available_names}"
    )


def _targon_failure_reason(*, app_id: str, state: Any, events: Any, logs: str) -> str | None:
    state_message = ""
    state_status = ""
    if isinstance(state, dict):
        state_status = str(state.get("status") or "").lower()
        state_message = str(state.get("message") or "")
    if state_status in {"failed", "error", "terminated", "stopped", "deleted"}:
        return f"Targon deployment {app_id} reached terminal status {state_status}: {state_message}".rstrip(": ")

    event_items = events.get("items", events) if isinstance(events, dict) else events
    fatal_events: list[str] = []
    if isinstance(event_items, list):
        for event in event_items:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("event_type") or "").upper()
            reason = str(event.get("reason") or "")
            message = str(event.get("display_message") or event.get("message") or "")
            exit_code = event.get("exit_code")
            is_fatal = (
                "CRASH" in event_type
                or "BACK_OFF" in event_type
                or "BACKOFF" in reason.upper()
                or (isinstance(exit_code, int) and exit_code != 0)
            )
            if is_fatal:
                suffix = f" (exit code {exit_code})" if isinstance(exit_code, int) else ""
                fatal_events.append(f"{event_type or reason}: {message}{suffix}")
    if fatal_events:
        log_excerpt = _targon_log_excerpt(logs)
        message = f"Targon deployment {app_id} failed while provisioning: {fatal_events[-1]}"
        if log_excerpt:
            message = f"{message}\nRecent logs:\n{log_excerpt}"
        return message

    if _logs_look_fatal(logs):
        return f"Targon deployment {app_id} failed while provisioning.\nRecent logs:\n{_targon_log_excerpt(logs)}"
    return None


def _logs_look_fatal(logs: str) -> bool:
    lowered = logs.lower()
    fatal_terms = ("traceback", "valueerror", "runtimeerror", "cuda out of memory", "exception")
    return any(term in lowered for term in fatal_terms)


def _targon_log_excerpt(logs: str, *, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in logs.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _deploy_targon_serverless_container(
    *,
    request: Any,
    app_name: str,
    resource: str,
    min_replicas: int,
    max_replicas: int,
    port: int,
    startup_timeout: int | None,
    requires_auth: bool,
    env: dict[str, str],
    vllm_args: list[str],
) -> dict[str, Any]:
    create_payload = {
        "type": "SERVERLESS",
        "name": app_name,
        "image": RUNPOD_VLLM_IMAGE_NAME,
        "resource_name": resource,
        "commands": ["sh", "-c"],
        "args": [_shell_join_with_runtime_port(["exec", "vllm", "serve", *vllm_args])],
        "envs": [{"name": str(name), "value": str(value)} for name, value in env.items()],
        "ports": [{"port": port, "protocol": "TCP"}],
        "serverless_config": {
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "container_concurrency": 100,
            "target_concurrency": 100,
            "visibility": "external",
            "requires_auth": requires_auth,
        },
    }
    if startup_timeout is not None:
        create_payload["serverless_config"]["startup_timeout"] = startup_timeout
        create_payload["serverless_config"]["webhook_config"] = {
            "type": "web_server",
            "port": port,
            "startup_timeout": startup_timeout,
            "requires_auth": requires_auth,
        }
    created = request("POST", "/tha/v2/workloads", json=create_payload)
    if not isinstance(created, dict) or not created.get("uid"):
        raise APIError("Targon did not return a workload id after creating the serverless container.")
    deployed = request("POST", f"/tha/v2/workloads/{created['uid']}/deploy")
    if isinstance(deployed, dict):
        deployed.setdefault("uid", created.get("uid"))
        deployed.setdefault("name", created.get("name"))
        if not deployed.get("envs"):
            deployed["envs"] = create_payload["envs"]
        return deployed
    created["envs"] = create_payload["envs"]
    return created


def _targon_app_from_workload(item: dict[str, Any]) -> TargonApp:
    state = item.get("state") if isinstance(item.get("state"), dict) else {}
    urls = state.get("urls") if isinstance(state.get("urls"), list) else item.get("urls")
    server_url = ""
    if isinstance(urls, list):
        for url_item in urls:
            if isinstance(url_item, dict) and url_item.get("url"):
                server_url = str(url_item["url"])
                break
    metadata = {}
    envs = item.get("envs")
    if isinstance(envs, list):
        metadata = {
            str(env.get("name")): str(env.get("value"))
            for env in envs
            if isinstance(env, dict) and env.get("name") is not None
        }
    return TargonApp(
        uid=str(item.get("uid") or item.get("id") or ""),
        name=str(item.get("name") or ""),
        status=str(state.get("status") or item.get("status") or ""),
        web_url=server_url,
        message=str(state.get("message") or item.get("message") or ""),
        metadata=metadata,
    )


def _targon_server_url_from_app(app: TargonApp) -> str:
    return app.web_url or app.endpoint_url or app.url or ""


def _deployment_key_from_targon_app(app: TargonApp) -> str:
    key = app.metadata.get("GRADIENTS_DEPLOYMENT_KEY")
    return str(key) if key else ""
