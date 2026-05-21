<img src="assets/banner-inference.svg" width="800" alt="Inference">

<br>

Inference in the SDK is vLLM-first. Local and RunPod deployments both expose an OpenAI-compatible server, so the same sampler can talk to either one.

Use `ModelSampler` when you want the SDK to start local vLLM for you. Use `RemoteVLLMSampler` when you already have a server URL.

For generation knobs such as `temperature`, `top_p`, `stop`, and penalties, see [Inference Parameters](inference/parameters.md).

<br>

<img src="assets/section-model-sampler.svg" width="800" alt="ModelSampler">

`ModelSampler` starts local vLLM by default, sends prompts to it, and stops the server after generation unless you ask it to keep the server alive.

> [!NOTE]
> Local vLLM inference requires a CUDA GPU and `pip install vllm`.

Generate with a base model:

```python
from gradientsio import ModelSampler

sampler = ModelSampler()
answers = sampler.generate("Qwen/Qwen2.5-3B", ["What is DNA?"])
print(answers[0])
```

Pass OpenAI-compatible request parameters directly:

```python
answers = sampler.generate(
    "Qwen/Qwen2.5-3B",
    ["What is DNA?"],
    max_tokens=128,
    temperature=0.2,
    top_p=0.95,
)
```

Generate with your trained adapter:

```python
answers = sampler.generate_with_adapter(
    "gradients-ai/your-trained-adapter",
    ["What is DNA?"],
    base_model_repo="Qwen/Qwen2.5-3B",
)
print(answers[0])
```

Keep the local vLLM server alive across calls:

```python
sampler = ModelSampler(keep_server_alive=True)
```

Pass vLLM deployment options through `vllm_kwargs`:

```python
sampler = ModelSampler(
    vllm_kwargs={
        "port": 8001,
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.96,
    }
)
```

Compare base and trained outputs:

```python
from gradientsio import load_dataset_rows

samples = load_dataset_rows("your-test-dataset", sample_size=5)
prompts = [s["instruction"] for s in samples]

base = sampler.generate("Qwen/Qwen2.5-3B", prompts)
trained = sampler.generate_with_adapter(
    "gradients-ai/your-trained-adapter",
    prompts,
    base_model_repo="Qwen/Qwen2.5-3B",
)

for s, b, t in zip(samples, base, trained):
    print(f"Q: {s['instruction'][:80]}...")
    print(f"Base:    {b[:120]}...")
    print(f"Trained: {t[:120]}...")
    print()
```

<br>

---

<img src="assets/section-generation-config.svg" width="800" alt="Generation config">

Control generation behavior with `GenerationConfig`:

```python
from gradientsio import GenerationConfig

config = GenerationConfig(
    max_new_tokens=256,
    do_sample=True,
    repetition_penalty=1.2,
    num_beams=1,
    max_input_tokens=4096,
)

answers = sampler.generate("Qwen/Qwen2.5-3B", prompts, config=config)
```

| Parameter | Default | Description |
|---|---|---|
| `max_new_tokens` | `96` | Maximum tokens to generate per prompt |
| `do_sample` | `False` | `True` for sampling, `False` for greedy/beam search |
| `repetition_penalty` | `1.12` | Penalizes repeated tokens. Higher = less repetition |
| `num_beams` | `4` | Beam search width. Set to `1` for greedy decoding |
| `max_input_tokens` | `3072` | Truncates input to this length |

For vLLM, `max_new_tokens` maps to `max_tokens`. If outputs are cut short, increase `max_new_tokens`. If they're repetitive, increase `repetition_penalty` or try `do_sample=True`.

<br>

---

<img src="assets/section-adapters.svg" width="800" alt="Working with adapters">

Gradients training produces LoRA adapters — small weight files that modify the base model's behavior without replacing it. The vLLM path serves the base model and applies the adapter at inference time.

If you need more control, load the model and adapter manually:

```python
tokenizer, model = sampler.load_model(
    "your-trained-model-repo",
    base_model_repo="Qwen/Qwen2.5-3B",
)

# run multiple prompts without reloading
answers = sampler.generate_with_model(tokenizer, model, prompts)
more_answers = sampler.generate_with_model(tokenizer, model, more_prompts)

# free GPU memory when done
ModelSampler.release_model(model)
```

This Transformers fallback avoids vLLM if you specifically need in-process model objects.

Use it by setting `backend="transformers"`:

```python
sampler = ModelSampler(backend="transformers")
```

<br>

---

<img src="assets/section-beyond-sampler.svg" width="800" alt="Beyond ModelSampler">

Deploy a server first when you want to reuse it or share it with another app. Provider-specific inference notes live in:

- **[Local vLLM Inference](inference/local-vllm.md)**
- **[RunPod vLLM Inference](inference/runpod-vllm.md)**

**Local vLLM**:

```python
import gradientsio

deployment = gradientsio.deploy_local_vllm(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-model-abc123",
)
```

**RunPod vLLM**:

```python
import gradientsio

deployment = gradientsio.deploy_runpod(
    base_model="Qwen/Qwen2.5-3B",
    lora="gradients-ai/your-model-abc123",
)
deployment.wait_ready(timeout=1800)

sampler = deployment.sampler()
answers = sampler.generate(["What is DNA?"], max_tokens=256)
```

Use an existing server URL:

```python
from gradientsio import RemoteVLLMSampler

sampler = RemoteVLLMSampler(
    base_url="http://127.0.0.1:8000",
    model="gradients-gradients-ai-your-model-abc123",
)

answers = sampler.generate(["What is DNA?"])
```

Take the result from any Gradients training job and plug it into deployment:

```python
# train your model
task = client.train(
    model="Qwen/Qwen2.5-3B",
    task_type=TaskType.INSTRUCT,
    hours=2,
    dataset="your-dataset",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)
result = task.wait()
trained_model = result.trained_model_repository  # e.g. "gradients-ai/your-model-abc123"
```

```python
deployment = gradientsio.deploy_runpod(
    base_model="Qwen/Qwen2.5-3B",
    lora=trained_model,
)
```

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Deployment](deployment.md)** — Overview of local and RunPod deployment.
- **[Inference Parameters](inference/parameters.md)** — OpenAI-compatible request options.
- **[Local vLLM Inference](inference/local-vllm.md)** — Sample from a local vLLM server.
- **[RunPod vLLM Inference](inference/runpod-vllm.md)** — Sample from a RunPod vLLM server.
- **[Scheduler](scheduler.md)** — Multi-iteration training across multiple datasets.
