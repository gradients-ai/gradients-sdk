<img src="assets/banner-inference.svg" width="800" alt="Inference">

<br>

Gradients trains your model and publishes it to Hugging Face. How you run inference is up to you — the output is a standard LoRA adapter or merged model, compatible with any serving stack.

The SDK includes `ModelSampler`, a lightweight utility for quick local testing. For production, you'll want a dedicated serving solution.

<br>

<img src="assets/section-model-sampler.svg" width="800" alt="ModelSampler">

`ModelSampler` is a convenience wrapper around Hugging Face Transformers for quick testing. It loads a model, runs prompts through it, and returns the outputs. It handles adapter detection and merging automatically.

> [!NOTE]
> `ModelSampler` requires a CUDA GPU and PyTorch with CUDA support. It's designed for testing, not production serving.

```python
from gradientsio import ModelSampler

sampler = ModelSampler()

# check GPU availability
print(sampler.cuda_status())
```

Generate with a base model:

```python
answers = sampler.generate("Qwen/Qwen2.5-3B", ["What is DNA?"])
print(answers[0])
```

Generate with your trained adapter, merged on-the-fly with the base model:

```python
answers = sampler.generate_with_adapter(
    "your-trained-model-repo",
    ["What is DNA?"],
    base_model_repo="Qwen/Qwen2.5-3B",
)
print(answers[0])
```

Compare both side by side:

```python
from gradientsio import load_dataset_rows

samples = load_dataset_rows("your-test-dataset", sample_size=5)
prompts = [s["instruction"] for s in samples]

base = sampler.generate("Qwen/Qwen2.5-3B", prompts)
trained = sampler.generate_with_adapter("your-trained-model-repo", prompts, base_model_repo="Qwen/Qwen2.5-3B")

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

For quick tests, the defaults are fine. If outputs are cut short, increase `max_new_tokens`. If they're repetitive, increase `repetition_penalty` or try `do_sample=True`.

<br>

---

<img src="assets/section-adapters.svg" width="800" alt="Working with adapters">

Gradients training produces LoRA adapters — small weight files that modify the base model's behavior without replacing it. `ModelSampler` detects adapters automatically and merges them on-the-fly.

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

This avoids reloading the model for each call — useful when testing across multiple prompt sets.

`load_model` auto-detects whether the repo contains a LoRA adapter (by checking for `adapter_config.json`) or a full model, and handles both cases.

<br>

---

<img src="assets/section-beyond-sampler.svg" width="800" alt="Beyond ModelSampler">

`ModelSampler` is for quick validation — it loads the model fresh each time and processes prompts sequentially. For anything beyond testing, use a proper serving stack.

Your trained model is published to Hugging Face as a standard LoRA adapter. It works with any tool that supports LoRA:

**RunPod deployment from the SDK** — provision an H100 vLLM pod and get an endpoint:

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

Take the result from any Gradients training job and plug it straight into your serving stack:

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

That `trained_model` repo is a standard LoRA adapter on Hugging Face. Use it with any of these:

**vLLM** — high-throughput serving with LoRA hot-loading:

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="Qwen/Qwen2.5-3B", enable_lora=True)

answers = llm.generate(
    prompts,
    SamplingParams(max_tokens=256),
    lora_request=LoRARequest("gradients-adapter", 1, trained_model),
)
```

**Text Generation Inference (TGI)** — deploy as a Docker container:

```bash
docker run --gpus all \
  -e MODEL_ID=Qwen/Qwen2.5-3B \
  -e LORA_ADAPTERS=gradients-ai/your-model-abc123 \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference
```

**Transformers + PEFT** — direct Python usage:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", device_map="auto")
model = PeftModel.from_pretrained(base, trained_model)
model = model.merge_and_unload()
```

**Merging permanently** — push a standalone model without the adapter dependency:

```python
tokenizer, model = sampler.load_model(trained_model, base_model_repo="Qwen/Qwen2.5-3B")
model.push_to_hub("your-org/merged-model")
tokenizer.push_to_hub("your-org/merged-model")
```

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Architecture](architecture.md)** — How the platform works under the hood — tournaments, validators, miners, and LoRA.
- **[API Reference](api-reference.md)** — Complete class, method, and type reference.
- **[Deployment](deployment.md)** — Deploy a trained LoRA adapter to RunPod with vLLM.
- **[Scheduler](scheduler.md)** — Multi-iteration training across multiple datasets.
