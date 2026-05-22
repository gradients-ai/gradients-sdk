# Basilica vLLM Inference

Basilica inference uses the same OpenAI-compatible sampler as local, RunPod, Lium, and Targon vLLM. The server URL comes from a Basilica deployment.

## Setup

Set your Basilica API key before deploying:

```bash
export BASILICA_API_KEY="your-basilica-api-key"
```

The Gradients SDK reads `BASILICA_API_KEY` for deployment creation, reuse, logs, and deletion.

## Deploy and Sample

```python
import gradientsio

deployment = gradientsio.deploy_basilica(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

deployment.wait_ready(timeout=1800)

sampler = deployment.sampler()
answers = sampler.generate(["What is DNA?"], max_tokens=128)
print(answers[0])
```

By default, Basilica deployments request A100 GPUs with at least 80 GB of GPU memory. Pass `gpu_models=[...]` or `min_gpu_memory_gb=...` to override this.

Tune the OpenAI-compatible request:

```python
answers = sampler.generate(
    ["What is DNA?"],
    max_tokens=128,
    temperature=0.2,
    top_p=0.95,
    stop=["\n\n"],
)
```

## Existing Basilica Server

If you already have a server URL:

```python
from gradientsio import RemoteVLLMSampler

sampler = RemoteVLLMSampler(
    base_url="https://your-basilica-deployment-url",
    model="gradients-gradients-ai-your-trained-adapter",
)

answers = sampler.generate(["What is DNA?"])
```

## Direct HTTP

```bash
curl "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-served-model-name",
    "prompt": "What is DNA?",
    "max_tokens": 128
  }'
```

See [Inference Parameters](parameters.md) for the full parameter table.
