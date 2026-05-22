<img src="assets/banner-deployments.svg" width="800" alt="Deployment">

<br>

Gradients publishes trained text models as standard Hugging Face LoRA adapters. The deployment helpers provision a serving endpoint for a base model plus adapter and return a handle you can use from the SDK or from any HTTP client.

Every deployment exposes an OpenAI-compatible vLLM server. That means the same `deployment.sampler()` call works whether the server is local, on RunPod, on Lium, on Targon, or on Basilica.

<br>

---

## Options

| Option | SDK call | Inference guide |
|---|---|---|
| [Local vLLM](#serve-locally) | `gradientsio.deploy_local_vllm(...)` | [Local vLLM inference](inference/local-vllm.md) |
| [RunPod](#serve-on-runpod) | `gradientsio.deploy_runpod(...)` | [RunPod vLLM inference](inference/runpod-vllm.md) |
| [Lium](#serve-on-lium) | `gradientsio.deploy_lium(...)` | [Lium vLLM inference](inference/lium-vllm.md) |
| [Targon](#serve-on-targon) | `gradientsio.deploy_targon(...)` | [Targon vLLM inference](inference/targon-vllm.md) |
| [Basilica](#serve-on-basilica) | `gradientsio.deploy_basilica(...)` | [Basilica vLLM inference](inference/basilica-vllm.md) |

All options return a deployment handle with:

- `deployment.server_url`
- `deployment.sampler()`
- `deployment.delete()`

<br>

---

## Quick E2E Flow

The deployment flow is the same across providers: deploy, wait until vLLM is ready, sample, then delete when you are done.

```python
import gradientsio

deployment = gradientsio.deploy_basilica(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

sampler = deployment.sampler()
answers = sampler.generate(
    ["What is DNA?"],
    max_tokens=128,
    temperature=0.2,
)
print(answers[0])

deployment.delete()
```

Each provider section below explains the setup and provider-specific options. Continue to the linked inference guide for provider-specific inference examples, existing-server usage, direct HTTP calls, and request parameters.

<br>

---

## Serve Locally

Local vLLM starts a server on your own GPU machine. Use this when you want direct control over the environment, fast iteration, or a local server that other code can call.

Install the GPU extra first:

```bash
pip install "gradientsio[gpu]"
```

Deploy locally:

```python
import gradientsio

deployment = gradientsio.deploy_local_vllm(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

print(deployment.server_url)
```

The default local server URL is `http://127.0.0.1:8000`. This binds to `127.0.0.1`, so it only accepts requests from the same machine. To accept external requests, use `host="0.0.0.0"` and connect through the machine's real IP or DNS name.

For local deployments, `deployment.delete()` stops the vLLM process started by the SDK. If the SDK reused an existing server, it leaves that server running.

Continue with [Local vLLM inference](inference/local-vllm.md).

<br>

---

## Serve On RunPod

[RunPod](https://www.runpod.io/) provides cloud GPU pods. The Gradients SDK creates a vLLM pod, waits for the OpenAI-compatible server to respond, and returns a deployment handle.

Set your RunPod API key before deploying:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

The RunPod API key is read only from `RUNPOD_API_KEY`. It is not accepted as a Python argument.

Deploy on RunPod:

```python
deployment = gradientsio.deploy_runpod(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

print(deployment.server_url)
```

The SDK logs the server URL once when it creates or reconnects to a RunPod deployment. `wait_ready()` checks `/v1/models` and waits for a healthy vLLM response:

```python
deployment.wait_ready(timeout=1800)
```

Leave `lora` unset to serve a base model only:

```python
deployment = gradientsio.deploy_runpod(base_model="Qwen/Qwen2.5-3B")
```

RunPod deployments are idempotent without a Gradients database or local cache. The SDK stores a deterministic key on the pod:

- `GRADIENTS_DEPLOYMENT_KEY`
- `GRADIENTS_BASE_MODEL`
- `GRADIENTS_LORA`
- `GRADIENTS_SDK_PROVIDER`

Before creating a pod, `deploy_runpod()` lists your RunPod pods and reuses a non-terminated pod with the same deployment key. The key includes the base model, LoRA adapter, vLLM image, port, GPU count, GPU type preferences, and vLLM startup command.

RunPod-only options include `template_id`, `gpu_type_ids`, `cloud_type`, `container_disk_in_gb`, `volume_in_gb`, and `interruptible`.

For RunPod deployments, `deployment.delete()` deletes the RunPod pod.

Continue with [RunPod vLLM inference](inference/runpod-vllm.md).

<br>

---

## Serve On Lium

[Lium](https://lium.io/) provides GPU pods from a marketplace of available machines. The Gradients SDK creates or reuses a vLLM template, rents a compatible executor, and exposes the pod as an OpenAI-compatible server.

Set your Lium API key before deploying:

```bash
export LIUM_API_KEY="your-lium-api-key"
```

The Lium API key is read only from `LIUM_API_KEY`. It is not accepted as a Python argument.

Lium's pod API requires an SSH public key. The SDK uses the first registered Lium SSH key. If none exists, it uses `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`; if no local key exists, it creates `~/.ssh/id_ed25519`, registers the public key with Lium.

Deploy on Lium:

```python
deployment = gradientsio.deploy_lium(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

print(deployment.server_url)
```

Lium deployments use a private Gradients-managed vLLM template and a rented Lium executor. The SDK creates the template when needed, rents a compatible GPU executor, waits for the Lium pod to reach `RUNNING`, then checks `/v1/models` for a healthy vLLM response.

```python
deployment.wait_ready(timeout=1800)
```

Leave `lora` unset to serve a base model only:

```python
deployment = gradientsio.deploy_lium(base_model="Qwen/Qwen2.5-3B")
```

Lium deployments are idempotent without a Gradients database or local cache. The SDK uses a deterministic deployment key and pod/template name derived from the base model, LoRA adapter, vLLM image, port, GPU count, environment, and vLLM startup command. If it finds a non-terminal Lium pod with the same key or name, it reuses that pod.

Lium-only options include `gpu_type` and `termination_hours`.

For Lium deployments, `deployment.delete()` deletes the Lium pod.

> [!WARNING]
> Do not put long-lived secrets on non-CVM Lium pods. GPU providers may be able to inspect container files, environment variables, and process memory on non-CVM machines. Prefer public models/adapters, short-lived tokens, or CVM nodes for sensitive workloads.

Continue with [Lium vLLM inference](inference/lium-vllm.md).

<br>

---

## Serve On Targon

[Targon](https://www.targon.com/) provides serverless GPU web endpoints. The Gradients SDK builds a vLLM workload around your model and returns the endpoint URL once the server is reachable.

Set your Targon API key before deploying:

```bash
export TARGON_API_KEY="your-targon-api-key"
```

The SDK reads `TARGON_API_KEY` for app reuse and deletion.

Deploy on Targon:

```python
deployment = gradientsio.deploy_targon(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

print(deployment.server_url)
```

Targon deployments use Targon Serverless web endpoints. The SDK builds a Targon app around `vllm serve`, deploys it with the requested GPU resource, then checks `/v1/models` for a healthy OpenAI-compatible response.

By default, Targon resources are tried in this order: `h100-small`, `h200-small`, `b200-small`, then `rtx4090-small`. Pass `resource="..."` to force a specific resource.

```python
deployment.wait_ready(timeout=1800)
```

Leave `lora` unset to serve a base model only:

```python
deployment = gradientsio.deploy_targon(base_model="Qwen/Qwen2.5-3B")
```

Targon deployments use a deterministic app name derived from the base model, LoRA adapter, Targon resource, port, environment, and vLLM startup command. If the SDK finds an existing Targon app with the same name, it reuses it.

Targon-only options include `resource`, `project_name`, `startup_timeout`, and `requires_auth`.

For Targon deployments, `deployment.delete()` deletes the Targon app.

Continue with [Targon vLLM inference](inference/targon-vllm.md).

<br>

---

## Serve On Basilica

[Basilica](https://www.basilica.ai/) provides developer-native compute for container deployments. The Gradients SDK uses the Basilica deployments API directly, requests A100 GPUs by default, and waits for vLLM to become reachable.

Set your Basilica API key before deploying:

```bash
export BASILICA_API_KEY="your-basilica-api-key"
```

The Basilica API key is read only from `BASILICA_API_KEY`. It is not accepted as a Python argument.

Deploy on Basilica:

```python
deployment = gradientsio.deploy_basilica(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

print(deployment.server_url)
```

Basilica deployments use the Basilica deployments API directly. The SDK creates a `vllm/vllm-openai:latest` deployment, requests A100 GPUs by default, waits for the deployment to become healthy, then checks `/v1/models` for a healthy OpenAI-compatible response.

By default, Basilica deployments request `gpu_models=["A100"]` and `min_gpu_memory_gb=80`. Pass `gpu_models=[...]` or `min_gpu_memory_gb=...` to override this.

```python
deployment.wait_ready(timeout=1800)
```

Leave `lora` unset to serve a base model only:

```python
deployment = gradientsio.deploy_basilica(base_model="Qwen/Qwen2.5-3B")
```

Basilica deployments use a deterministic deployment name derived from the base model, LoRA adapter, vLLM image, port, GPU settings, environment, and vLLM startup command. If the SDK finds an existing non-terminal Basilica deployment with the same name or deployment key, it reuses it.

Basilica-only options include `gpu_models`, `min_gpu_memory_gb`, `cpu`, `memory`, and `ttl_seconds`.

For Basilica deployments, `deployment.delete()` deletes the Basilica deployment.

Continue with [Basilica vLLM inference](inference/basilica-vllm.md).

<br>

---

## Configure vLLM

The local, RunPod, Lium, Targon, and Basilica helpers accept the same core vLLM settings:

```python
deployment = gradientsio.deploy_local_vllm(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora="gradients-ai/your-trained-adapter",
    port=8001,
    max_model_len=8192,
    gpu_memory_utilization=0.96,
    dtype="bfloat16",
    trust_remote_code=True,
    enforce_eager=False,
    max_lora_rank=256,
    gpu_count=1,
)
```

Common options:

| Parameter | Default | Description |
|---|---|---|
| `base_model` | required | Hugging Face base model repo |
| `lora` | `None` | Optional Hugging Face LoRA adapter repo |
| `deployment_model_name` | derived from served repo | Model name used in vLLM requests |
| `hf_token` | `None` | Hugging Face token for private or gated repos |
| `port` | `8000` | vLLM HTTP port |
| `gpu_count` | inferred for cloud providers, `1` locally | Number of GPUs; also sets vLLM tensor parallelism |
| `max_model_len` | inferred from model size | vLLM `--max-model-len` |
| `gpu_memory_utilization` | `0.96` | vLLM `--gpu-memory-utilization` |
| `dtype` | vLLM default | vLLM `--dtype` |
| `trust_remote_code` | `False` | Adds vLLM `--trust-remote-code` |
| `enforce_eager` | `False` | Adds vLLM `--enforce-eager` |
| `max_lora_rank` | `256` | vLLM `--max-lora-rank` for LoRA adapters |

The SDK adds flags only when they are needed or provided. For example, `max_model_len=8192` becomes `--max-model-len 8192`, `gpu_memory_utilization=0.90` becomes `--gpu-memory-utilization 0.90`, `dtype="bfloat16"` becomes `--dtype bfloat16`, `trust_remote_code=True` adds `--trust-remote-code`, and `enforce_eager=True` adds `--enforce-eager`.

For larger models, start with explicit memory settings:

```python
deployment = gradientsio.deploy_local_vllm(
    base_model="Qwen/Qwen2.5-Coder-32B-Instruct",
    lora="gradients-ai/your-trained-adapter",
    gpu_count=2,
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    dtype="bfloat16",
    trust_remote_code=True,
    enforce_eager=True,
)
```

If vLLM fails with a KV cache or CUDA out-of-memory error:

- Lower `max_model_len` first. Long context lengths reserve more KV cache memory, even for short prompts.
- Lower `gpu_memory_utilization` if the GPU is shared or fragmented. Values like `0.85` to `0.92` are safer than filling the card.
- Increase `gpu_count` for larger models. The SDK maps this to vLLM `--tensor-parallel-size`.
- Use `dtype="bfloat16"` or `dtype="float16"` for models that support it.
- Try `enforce_eager=True` if CUDA graph capture increases memory pressure or startup fails during graph capture. It can use less peak memory at the cost of some throughput.
- Close other GPU processes before starting vLLM. Check with `nvidia-smi`.
- Serve the base model without `lora` first to confirm the model fits, then add the adapter.
- Keep `max_lora_rank=256` unless your adapter requires a larger rank. Larger ranks use more memory.

<br>

---

## Full Inference Example

This example uses Basilica, but the sampler calls are the same for every provider.

```python
import gradientsio

deployment = gradientsio.deploy_basilica(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    dtype="bfloat16",
)

try:
    deployment.wait_ready(timeout=1800, poll_interval=20)

    sampler = deployment.sampler(timeout=120)
    answers = sampler.generate(
        [
            "Explain what a LoRA adapter is in one paragraph.\n\nAnswer:",
            "What is DNA?\n\nAnswer:",
        ],
        max_tokens=160,
        temperature=0.2,
        top_p=0.95,
    )

    for answer in answers:
        print(answer)
finally:
    deployment.delete()
```

You can also call the server directly if you do not want to use the SDK sampler:

```bash
curl "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-served-model-name",
    "prompt": "What is DNA?",
    "max_tokens": 128
  }'
```

For all generation parameters, see [Inference Parameters](inference/parameters.md).

<br>

---

## What To Read Next

- **[Inference](inference.md)** — Sampling with local or cloud OpenAI-compatible servers.
- **[Inference Parameters](inference/parameters.md)** — Request parameters such as `temperature`, `top_p`, `stop`, and penalties.
