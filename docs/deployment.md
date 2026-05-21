<img src="assets/banner-deployments.svg" width="800" alt="Deployment">

<br>

Gradients publishes trained text models as standard Hugging Face LoRA adapters. The deployment helpers provision a serving endpoint for a base model plus adapter and return a handle you can use from the SDK or from any HTTP client.

Currently we can deploy a local vLLM server, or a RunPod vLLM server.

<br>

---

## Options

| Option | SDK call |
|---|---|
| Local vLLM | `gradientsio.deploy_local_vllm(...)` |
| RunPod vLLM | `gradientsio.deploy_runpod(...)` |

Both options return a deployment handle with:

- `deployment.server_url`
- `deployment.sampler()`
- `deployment.delete()`

<br>

---

## Setup

Install the SDK:

```bash
pip install gradientsio
```

For local vLLM, install vLLM in the same environment:

```bash
pip install vllm
```

For RunPod, set your RunPod API key before calling the SDK:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

The RunPod API key is read only from `RUNPOD_API_KEY`. It is not accepted as a Python argument.

For private or gated Hugging Face repos, set a token or pass `hf_token`:

```bash
export HF_TOKEN="hf_..."
```

<br>

---

## Serve Locally

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

<br>

---

## Serve On RunPod

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
deployment = gradientsio.deploy_local_vllm(base_model="Qwen/Qwen2.5-3B")
```

RunPod deployments are idempotent without a Gradients database or local cache. The SDK stores a deterministic key on the pod:

- `GRADIENTS_DEPLOYMENT_KEY`
- `GRADIENTS_BASE_MODEL`
- `GRADIENTS_LORA`
- `GRADIENTS_SDK_PROVIDER`

Before creating a pod, `deploy_runpod()` lists your RunPod pods and reuses a non-terminated pod with the same deployment key. The key includes the base model, LoRA adapter, vLLM image, port, GPU count, GPU type preferences, and vLLM startup command.

For RunPod deployments, `deployment.delete()` deletes the RunPod pod.

<br>

---

## Configure vLLM

The local and RunPod helpers accept the same core vLLM settings:

```python
deployment = gradientsio.deploy_local_vllm(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora="gradients-ai/your-trained-adapter",
    port=8001,
    max_model_len=8192,
    gpu_memory_utilization=0.96,
    dtype="bfloat16",
    trust_remote_code=True,
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
| `gpu_count` | inferred for RunPod, `1` locally | Number of GPUs; also sets vLLM tensor parallelism |
| `max_model_len` | inferred from model size | vLLM `--max-model-len` |
| `gpu_memory_utilization` | `0.96` | vLLM `--gpu-memory-utilization` |
| `dtype` | vLLM default | vLLM `--dtype` |
| `trust_remote_code` | `False` | Adds vLLM `--trust-remote-code` |
| `max_lora_rank` | `256` | vLLM `--max-lora-rank` for LoRA adapters |

RunPod-only options include `template_id`, `gpu_type_ids`, `cloud_type`, `container_disk_in_gb`, `volume_in_gb`, and `interruptible`.

<br>

---

## Sample From Any Deployment

```python
sampler = deployment.sampler()
answers = sampler.generate(["What is DNA?"], max_tokens=128)
print(answers[0])
```

Or call the server directly:

```bash
curl "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-served-model-name",
    "prompt": "What is DNA?",
    "max_tokens": 128
  }'
```

<br>

---

## What To Read Next

- **[Inference](inference.md)** — Sampling with local or cloud OpenAI-compatible servers.
