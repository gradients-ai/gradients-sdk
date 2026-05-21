# Lium vLLM Inference

Lium inference uses the same OpenAI-compatible sampler as local and RunPod vLLM. The difference is that the server URL comes from a Lium pod.

## Setup

Set your Lium API key before deploying:

```bash
export LIUM_API_KEY="your-lium-api-key"
```

Lium's pod API requires an SSH public key. The SDK uses the first registered Lium SSH key. If none exists, it uses `~/.ssh/id_ed25519.pub` or `~/.ssh/id_rsa.pub`; if no local key exists, it creates `~/.ssh/id_ed25519`, registers the public key with Lium.

## Deploy and Sample

```python
import gradientsio

deployment = gradientsio.deploy_lium(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

deployment.wait_ready(timeout=1800)

sampler = deployment.sampler()
answers = sampler.generate(["What is DNA?"], max_tokens=128)
print(answers[0])
```

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

## Existing Lium Server

If you already have a server URL:

```python
from gradientsio import RemoteVLLMSampler

sampler = RemoteVLLMSampler(
    base_url="http://your-lium-host:your-forwarded-port",
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

> [!WARNING]
> Do not place long-lived secrets on non-CVM Lium pods. Use public models/adapters, short-lived tokens, or CVM nodes for sensitive workloads.

See [Inference Parameters](parameters.md) for the full parameter table.
