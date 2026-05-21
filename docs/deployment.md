<img src="assets/banner-deployments.svg" width="800" alt="Deployment">

<br>

Gradients publishes trained text models as standard Hugging Face LoRA adapters. The deployment helpers provision a serving endpoint for a base model plus adapter and return a handle you can use from the SDK or from any HTTP client.

Currently we can deploy a local vLLM server, a RunPod vLLM server, or a Lium vLLM server.

<br>

---

## Options

| Option | SDK call |
|---|---|
| Local vLLM | `gradientsio.deploy_local_vllm(...)` |
| RunPod vLLM | `gradientsio.deploy_runpod(...)` |
| Lium vLLM | `gradientsio.deploy_lium(...)` |

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

For local vLLM, install the GPU extra in the same environment:

```bash
pip install "gradientsio[gpu]"
```

From a local checkout:

```bash
pip install -e ".[gpu]"
```

For RunPod, set your RunPod API key before calling the SDK:

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

The RunPod API key is read only from `RUNPOD_API_KEY`. It is not accepted as a Python argument.

For Lium, set your Lium API key before calling the SDK:

```bash
export LIUM_API_KEY="your-lium-api-key"
```

The Lium API key is read only from `LIUM_API_KEY`. It is not accepted as a Python argument.
Lium's pod API requires an SSH public key. The SDK uses the first registered Lium SSH key. If none exists, it uses `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`; if no local key exists, it creates `~/.ssh/id_ed25519`, registers the public key with Lium.

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

## Serve On Lium

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

For Lium deployments, `deployment.delete()` deletes the Lium pod.

> [!WARNING]
> Do not put long-lived secrets on non-CVM Lium pods. GPU providers may be able to inspect container files, environment variables, and process memory on non-CVM machines. Prefer public models/adapters, short-lived tokens, or CVM nodes for sensitive workloads.

<br>

---

## Configure vLLM

The local, RunPod, and Lium helpers accept the same core vLLM settings:

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

RunPod-only options include `template_id`, `gpu_type_ids`, `cloud_type`, `container_disk_in_gb`, `volume_in_gb`, and `interruptible`.

Lium-only options include `gpu_type` and `termination_hours`.

For local serving, the SDK starts vLLM with the same OpenAI-compatible server used by vLLM directly:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --served-model-name Qwen-Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules your-adapter=gradients-ai/your-trained-adapter \
  --max-lora-rank 256 \
  --enforce-eager \
  --tensor-parallel-size 1
```

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
