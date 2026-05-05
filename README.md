# Gradients Python SDK

Python SDK for launching and monitoring Gradients.io training jobs.

## Installation

```bash
pip install gradients-sdk
export GRADIENTS_API_KEY="..."
```

## Quick Start

```python
from gradients import GradientsClient

client = GradientsClient()

task = client.tasks.create_instruct(
    ds_repo="yahma/alpaca-cleaned",
    model_repo="Qwen/Qwen2.5-7B-Instruct",
    file_format="hf",
    hours_to_complete=1,
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)

result = task.wait(poll_interval=600)
print(result.trained_model_repository)
```

## Design

Gradients is a job-based training platform. The SDK mirrors that lifecycle:

1. Configure API-key auth.
2. Check pricing.
3. Create a task.
4. Persist the returned `task_id`.
5. Poll status until `success` or a failure state.
6. Fetch the trained model repository and optional miner breakdown.

The SDK intentionally does not perform registration, funding, or automatic paid retries.

## Main Resources

```python
client.tasks.create_instruct(...)
client.tasks.create_chat(...)
client.tasks.create_dpo(...)
client.tasks.create_grpo(...)
client.tasks.create_image(...)
client.tasks.get(task_id)
client.tasks.breakdown(task_id)

client.scheduler.create_job(...)
client.scheduler.results(job_id)

client.grpo.list_reward_functions()
client.grpo.add_reward_function(...)

client.network.status()
client.chutes.deploy(model_id="...", lora_id="...")
```

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `GRADIENTS_API_KEY` | Gradients API key | None |
| `GRADIENTS_API_URL` | API base URL | `https://api.gradients.io` |
