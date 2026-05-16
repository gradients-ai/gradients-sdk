<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); border-radius: 16px; padding: 48px 44px; margin-bottom: 8px;">
  <div style="display: flex; align-items: center; gap: 28px; font-family: system-ui, -apple-system, sans-serif;">
    <img src="../examples/logo.png" alt="Gradients" style="height: 88px;">
    <div>
      <h1 style="color: #fff; margin: 0; font-size: 2.2em; letter-spacing: -0.01em;">Getting Started</h1>
      <p style="color: #a8b2d1; margin: 10px 0 0 0; font-size: 1.15em; line-height: 1.5;">From zero to a fine-tuned model. Install the SDK, train on your data, and test the result.</p>
    </div>
  </div>
</div>

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Install the SDK</div>
</div>

```bash
pip install gradientsio
```

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Create an account</div>
</div>

Sign up at [gradients.io](https://www.gradients.io/) — one-click account creation. From your dashboard, generate an API key and fund your account with TAO.

```bash
export GRADIENTS_API_KEY="your-api-key"
```

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Pricing</div>
  <div style="color: #6b7280; font-size: 0.95em; margin-top: 6px; line-height: 1.5;">Pay per hour of training. The rate depends on model size.</div>
</div>

| Model size | Hourly rate |
|---|---|
| Up to 1B parameters | $10 / hr |
| Up to 7B parameters | $15 / hr |
| Up to 40B parameters | $25 / hr |
| 40B+ parameters | $50 / hr |
| Image models | $5 / hr |

No hidden fees. You can check the exact cost of a job before you run it:

```python
from gradientsio import GradientsClient

client = GradientsClient()

quote = client.tasks.check_text_price(
    model_repo="Qwen/Qwen2.5-3B",
    hours_to_complete=2,
)
print(quote.total_price)
```

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Train a model</div>
  <div style="color: #6b7280; font-size: 0.95em; margin-top: 6px; line-height: 1.5;">Pick a base model from Hugging Face and a dataset. Gradients handles everything else — data preparation, GPU allocation, training, evaluation, and publishing your fine-tuned model.</div>
</div>

```python
from gradientsio import GradientsClient, TaskType

client = GradientsClient()

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

This returns immediately with a task handle. Training runs remotely.

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Monitor progress</div>
</div>

Check the status of your training job:

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

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Test the result</div>
  <div style="color: #6b7280; font-size: 0.95em; margin-top: 6px; line-height: 1.5;">Use <code>ModelSampler</code> to compare the base model against your trained model locally. Requires a GPU.</div>
</div>

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

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">Task lifecycle</div>
  <div style="color: #6b7280; font-size: 0.95em; margin-top: 6px; line-height: 1.5;">Every training job moves through these states:</div>
</div>

```
PENDING → PREPARING_DATA → LOOKING_FOR_NODES → READY → TRAINING → EVALUATING → SUCCESS
```

If something goes wrong, the task moves to a failure state instead: `PREP_TASK_FAILURE`, `FAILURE_FINDING_NODES`, or `FAILURE`. Check `task.refresh().status` at any point.

<div style="border-left: 4px solid #7c3aed; padding: 20px 24px; margin: 36px 0 14px 0; background: linear-gradient(90deg, #f5f3ff 0%, #ffffff 100%); border-radius: 0 12px 12px 0; font-family: system-ui, -apple-system, sans-serif;">
  <div style="font-weight: 700; font-size: 1.2em; color: #1a1a2e;">What to read next</div>
</div>

- **[Task Types](task-types.md)** — Instruct is one of five training modes. Learn when to use Chat, DPO, GRPO, or Image training.
- **[Datasets](datasets.md)** — How to prepare your own data, supported formats, and field mappings.
- **[Configuration](configuration.md)** — All the parameters you can control and what they do.
