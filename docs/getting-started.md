<img src="assets/banner-getting-started.svg" width="800" alt="Getting Started">

<br>

<img src="assets/section-install-the-sdk.svg" width="800" alt="Install the SDK">

One package. Everything you need to train and test models.

```bash
pip install gradientsio
```

<br>

<img src="assets/section-create-an-account.svg" width="800" alt="Create an account">

Sign up, grab an API key, and fund your account — all from the dashboard.

Create your account at [gradients.io](https://www.gradients.io/) — one-click sign-up. From your dashboard, generate an API key and fund your account with TAO.

```bash
export GRADIENTS_API_KEY="your-api-key"
```

```python
from gradientsio import GradientsClient, TaskType

client = GradientsClient()
```

<br>

<img src="assets/section-pricing.svg" width="800" alt="Pricing">

Pay per hour of training. The rate depends on model size.

| Model size | Hourly rate | | |
|---|---|---|---|
| Up to 1B parameters | $10 / hr | 40B+ parameters | $50 / hr |
| Up to 7B parameters | $15 / hr | Image models | $5 / hr |
| Up to 40B parameters | $25 / hr | | |

If in doubt, you can check the exact cost of a job before you run it:

```python
quote = client.tasks.check_text_price(
    model_repo="Qwen/Qwen2.5-3B",
    hours_to_complete=2,
)
print(quote.total_price)
```

<br>

<img src="assets/section-train-a-model.svg" width="800" alt="Train a model">

Pick a base model from Hugging Face and a dataset. Gradients handles everything else — data preparation, GPU allocation, training, evaluation, and publishing.

```python
task = client.train(
    model="Qwen/Qwen2.5-3B",
    task_type=TaskType.INSTRUCT,
    hours=2,
    dataset="gradients-io-tournaments/PubMedQA-Normalized-Train",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)
```

- `model` — any Hugging Face model you want to fine-tune.
- `task_type` — the training mode. Instruct is for question/answer data, but there are [other modes](task-types.md) for conversations, preference pairs, and images.
- `hours` — how long to train.
- `dataset` — a Hugging Face dataset, or your own via S3.
- `field_*` — tells Gradients which columns in your dataset map to what. Here, the dataset has columns called `instruction`, `input`, and `output`.

For the full list of parameters, see [Configuration](configuration.md).

The call returns immediately with a task handle. Training runs remotely.

<br>

<img src="assets/section-monitor-progress.svg" width="800" alt="Monitor progress">

Check in on your job, or just wait for it to finish.

```python
details = task.refresh()
print(details.status)
```

Or block until it finishes:

```python
result = task.wait()
print(result.trained_model_repository)
```

The `trained_model_repository` is a Hugging Face repo containing your fine-tuned model. Training typically takes 1–3 hours depending on the model size and `hours` parameter.

If you close your session and come back later, reconnect to an existing task by ID:

```python
task = client.tasks.handle("your-task-id")
result = task.wait()
```

<br>

<img src="assets/section-test-the-result.svg" width="800" alt="Test the result">

Compare the base model against your trained model on held-out data.

> [!NOTE]
> Local inference with `ModelSampler` requires a GPU.

```python
from gradientsio import ModelSampler, load_dataset_rows

sampler = ModelSampler()

# load a few test questions
samples = load_dataset_rows("gradients-io-tournaments/PubMedQA-Normalized-Test")
prompts = [f"{s['instruction']}\n\nAnswer:" for s in samples]

# generate with both models
base_answers = sampler.generate("Qwen/Qwen2.5-3B", prompts)
trained_answers = sampler.generate_with_adapter(
    result.trained_model_repository,
    prompts,
    base_model_repo="Qwen/Qwen2.5-3B",
)

# compare
for sample, base, trained in zip(samples, base_answers, trained_answers):
    print(f"Question: {sample['instruction'][:100]}...")
    print(f"Expected: {sample['output'][:100]}...")
    print(f"Base:     {base[:100]}...")
    print(f"Trained:  {trained[:100]}...")
    print()
```

<br>

<img src="assets/section-task-lifecycle.svg" width="800" alt="Task lifecycle">

Every training job moves through these states:

```
PENDING → PREPARING_DATA → LOOKING_FOR_NODES → READY → TRAINING → EVALUATING → SUCCESS
```

For details on failure states and error handling, see [Configuration](configuration.md).

<br>

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Task Types](task-types.md)** — Instruct is one of five training modes. If you have conversations, use Chat. If you have preference pairs, use DPO. If you want reward-driven training, use GRPO. If you're working with images, there's a mode for that too.
- **[Datasets](datasets.md)** — The PubMedQA example used a Hugging Face dataset, but you can bring your own data in JSON, CSV, or via S3. This guide covers how to prepare it.
- **[Configuration](configuration.md)** — Everything you can control: hours, backends, pricing, polling, error handling, and what each parameter actually does.
