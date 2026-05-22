# Targon vLLM Inference

Targon inference uses the same OpenAI-compatible sampler as local, RunPod, and Lium vLLM. The server URL comes from a Targon Serverless web endpoint.

## Setup

Install Targon support and set your API key:

```bash
pip install "gradientsio[targon]"
export TARGON_API_KEY="your-targon-api-key"
```

Targon also supports `targon setup` for CLI/SDK credentials. The Gradients SDK reads `TARGON_API_KEY` for app reuse and deletion.

## Deploy and Sample

```python
import gradientsio

deployment = gradientsio.deploy_targon(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

deployment.wait_ready(timeout=1800)

sampler = deployment.sampler()
answers = sampler.generate(["What is DNA?"], max_tokens=128)
print(answers[0])
```

By default, Targon resources are tried in this order: `h100-small`, `h200-small`, `b200-small`, then `rtx4090-small`. Pass `resource="..."` to force a specific resource.

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

## Existing Targon Server

If you already have a server URL:

```python
from gradientsio import RemoteVLLMSampler

sampler = RemoteVLLMSampler(
    base_url="https://your-targon-endpoint",
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
