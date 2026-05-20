<img src="assets/banner-inference.svg" width="800" alt="Deployment">

<br>

Gradients publishes trained text models as standard Hugging Face LoRA adapters. The deployment helpers provision a serving endpoint for a base model plus adapter and return a handle you can use from the SDK or from any HTTP client.

The first supported provider is RunPod with the official vLLM template on H100 GPUs.

<br>

---

## RunPod

If you intend to deploy on RunPod, set your RunPod API key before calling the SDK. The SDK reads the RunPod API key from `RUNPOD_API_KEY`; it is not accepted as a Python argument. The default vLLM image is `vllm/vllm-openai:latest`.

```bash
export RUNPOD_API_KEY="your-runpod-api-key"
```

Deploy with the base model and trained model repo.

```python
import gradientsio

deployment = gradientsio.deploy_runpod(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
    hf_token="hf_...",  # only needed for private or gated repos
)

deployment.wait_ready(timeout=1800)

print(deployment.server_url)
```

The SDK logs the server URL when it creates or reconnects to a RunPod deployment, so it is visible even if you are using the deployment helper in a script.

The RunPod deployment uses vLLM's OpenAI-compatible HTTP API. You can use the returned URL directly:

```bash
curl "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gradients-gradients-ai-your-trained-adapter",
    "prompt": "What is DNA?",
    "max_tokens": 128
  }'
```

Or use the SDK sampler:

```python
sampler = deployment.sampler()
answers = sampler.generate(["What is DNA?"], max_tokens=128)
print(answers[0])
```

Clean up the pod when you are done:

```python
deployment.delete()
```

<br>

---

## Idempotency

The SDK does not need a Gradients database or local cache to avoid duplicate pods. It stores a deterministic deployment key on the RunPod pod:

- `GRADIENTS_DEPLOYMENT_KEY`
- `GRADIENTS_BASE_MODEL`
- `GRADIENTS_LORA`
- `GRADIENTS_SDK_PROVIDER`

Before creating a pod, `deploy_runpod()` lists your RunPod pods and returns the existing non-terminated pod with the same deployment key. The key is based on the provider, base model, LoRA adapter, template ID, port, GPU choice, and vLLM config.

<br>

---

## Configuration

Common options:

| Parameter | Default | Description |
|---|---|---|
| `base_model` | required | Hugging Face base model repo used when `lora` is an adapter |
| `lora` | required | Hugging Face repo for the trained output; may be a LoRA adapter or standalone model |
| `template_id` | `vllm/vllm-openai:latest` | RunPod vLLM image name |
| `deployment_model_name` | derived from deployed repo | Model name used in vLLM requests |
| `hf_token` | `None` | Hugging Face token passed to the pod |
| `port` | `8000` | Internal vLLM HTTP port exposed through the RunPod proxy |
| `gpu_type_ids` | H100 variants | RunPod GPU types to rent by availability |
| `gpu_count` | model-size based | Number of GPUs to attach; also sets vLLM `--tensor-parallel-size` when greater than 1 |
| `max_model_len` | model-size based | vLLM `--max-model-len` |
| `gpu_memory_utilization` | `0.96` | vLLM `--gpu-memory-utilization` |
| `dtype` | vLLM default | vLLM `--dtype` |
| `trust_remote_code` | `False` | Adds vLLM `--trust-remote-code` |
| `max_lora_rank` | `256` | vLLM `--max-lora-rank` when serving a LoRA adapter |
| `env` | `{}` | Extra vLLM environment variables; explicit arguments above take precedence |

Example with vLLM options:

```python
deployment = gradientsio.deploy_runpod(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora="gradients-ai/your-trained-adapter",
    max_model_len=8192,
    gpu_memory_utilization=0.96,
    dtype="bfloat16",
    trust_remote_code=True,
    max_lora_rank=256,
    gpu_count=1,
)
```

<br>

---

## What To Read Next

- **[Inference](inference.md)** — Local testing and serving options.
- **[Configuration](configuration.md)** — SDK environment variables and error handling.
- **[Scheduler](scheduler.md)** — Multi-iteration training across multiple datasets.
