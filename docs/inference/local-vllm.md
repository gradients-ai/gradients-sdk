# Local vLLM Inference

Use local vLLM inference when you want the SDK to start an OpenAI-compatible server on your machine and sample through it.

## Start Through ModelSampler

`ModelSampler` uses local vLLM by default:

```python
from gradientsio import ModelSampler

sampler = ModelSampler()
answers = sampler.generate("Qwen/Qwen2.5-3B", ["What is DNA?"])
print(answers[0])
```

Tune the OpenAI-compatible request:

```python
answers = sampler.generate(
    "Qwen/Qwen2.5-3B",
    ["What is DNA?"],
    max_tokens=128,
    temperature=0.2,
    top_p=0.95,
)
```

For a LoRA adapter:

```python
answers = sampler.generate_with_adapter(
    "gradients-ai/your-trained-adapter",
    ["What is DNA?"],
    base_model_repo="Qwen/Qwen2.5-3B",
)
```

Keep the server alive across calls:

```python
sampler = ModelSampler(keep_server_alive=True)
```

## Start a Server First

```python
import gradientsio

deployment = gradientsio.deploy_local_vllm(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-trained-adapter",
)

answers = deployment.sampler().generate(["What is DNA?"])
```

The default URL is `http://127.0.0.1:8000`. Use `host="0.0.0.0"` if other machines need to reach the server, then connect through the machine's real IP or DNS name.

## Existing Server

```python
from gradientsio import RemoteVLLMSampler

sampler = RemoteVLLMSampler(
    base_url="http://127.0.0.1:8000",
    model="gradients-gradients-ai-your-trained-adapter",
)

answers = sampler.generate(["What is DNA?"])
```

See [Inference Parameters](parameters.md) for the full parameter table.
