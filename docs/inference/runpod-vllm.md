# RunPod vLLM Inference

RunPod inference uses the same OpenAI-compatible sampler as local vLLM. The difference is that the server URL comes from a RunPod pod.

## Deploy and Sample

```python
import gradientsio

deployment = gradientsio.deploy_runpod(
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

## Existing RunPod Server

If you already have a server URL:

```python
from gradientsio import RemoteVLLMSampler

sampler = RemoteVLLMSampler(
    base_url="https://your-pod-8000.proxy.runpod.net",
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
