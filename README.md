<img src="docs/assets/banner-readme.svg" width="800" alt="Gradients SDK">

<br>

Foundation models like Llama, Qwen, and Stable Diffusion are powerful but generic. Post-training is how you make them yours — teaching a model your data, your domain, your task.

That usually means training scripts, hyperparameter tuning, and GPU infrastructure. Gradients replaces all of it with a single API call. Choose a model and a dataset — Gradients automatically optimizes the training configuration for your problem and delivers a production-ready model.

<br>

<img src="docs/assets/section-quick-start.svg" width="800" alt="Quick start">

```bash
pip install gradientsio
```

Sign up at [gradients.io](https://www.gradients.io/), grab an API key, and:

```python
import os
from gradientsio import GradientsClient, TaskType

os.environ["GRADIENTS_API_KEY"] = "your-api-key"

result = GradientsClient().train(
    model="Qwen/Qwen2.5-3B",
    task_type=TaskType.INSTRUCT,
    hours=2,
    dataset="yahma/alpaca-cleaned",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
).wait()

print(result.trained_model_repository)
```

Five training modes: `INSTRUCT`, `CHAT`, `DPO`, `GRPO`, `IMAGE`.

<br>

<img src="docs/assets/section-pricing.svg" width="800" alt="Pricing">

| Model size | Hourly rate | | |
|---|---|---|---|
| Up to 1B parameters | $10 / hr | 40B+ parameters | $50 / hr |
| Up to 7B parameters | $15 / hr | Image models | $5 / hr |
| Up to 40B parameters | $25 / hr | | |

<br>

<img src="docs/assets/section-documentation.svg" width="800" alt="Documentation">

<table>
  <tr>
    <td><strong><a href="docs/getting-started.md">Getting Started</a></strong></td>
    <td>Install, create an account, train your first model, and test the result.</td>
  </tr>
  <tr>
    <td><strong><a href="docs/task-types.md">Task Types</a></strong></td>
    <td>Instruct, Chat, DPO, GRPO, and Image — when to use each and how.</td>
  </tr>
  <tr>
    <td><strong><a href="docs/datasets.md">Datasets</a></strong></td>
    <td>How to prepare your data, with good and bad examples for each type.</td>
  </tr>
  <tr>
    <td><strong><a href="docs/configuration.md">Configuration</a></strong></td>
    <td>Every parameter, error handling, polling, and environment variables.</td>
  </tr>
  <tr>
    <td><strong><a href="docs/scheduler.md">Scheduler</a></strong></td>
    <td>Multi-iteration training for datasets larger than 300k rows.</td>
  </tr>
  <tr>
    <td><strong><a href="docs/inference.md">Inference</a></strong></td>
    <td>Test locally or deploy with vLLM, TGI, or Transformers.</td>
  </tr>
  <tr>
    <td><strong><a href="docs/account.md">Account</a></strong></td>
    <td>Billing, balance, and pricing.</td>
  </tr>
</table>

<br>

See the [training comparison notebook](examples/training_comparison.ipynb) for a live demo — train a medical QA model and compare it against the base model side by side.
