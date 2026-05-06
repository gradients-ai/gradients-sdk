# gradientsio Python SDK

Python SDK for launching and monitoring Gradients.io training jobs.

## Installation

```bash
pip install gradientsio
export GRADIENTS_API_KEY="..."
```

## Quick Start

```python
from gradientsio import GradientsClient
from gradientsio import TaskType

client = GradientsClient()

task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.INSTRUCT,
    hours=1,
    dataset="yahma/alpaca-cleaned",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)

result = task.wait(poll_interval=600)
print(result.trained_model_repository)
```

For prepared JSON/S3 datasets:

```python
from gradientsio import Datasets
from gradientsio import GradientsClient
from gradientsio import TaskType

client = GradientsClient()

task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.INSTRUCT,
    hours=1,
    dataset=Datasets.S3("https://example.com/train.json", test_data="https://example.com/test.json"),
)
```

Supported task types are available from `TaskType`:

```python
TaskType.INSTRUCT
TaskType.CHAT
TaskType.DPO
TaskType.GRPO
TaskType.IMAGE
```

## Design

Gradients is a job-based training platform. The SDK mirrors that lifecycle:

1. Configure API-key auth.
2. Check pricing.
3. Create a task.
4. Persist the returned `task_id`.
5. Poll status until `success` or a failure state.
6. Fetch the trained model repository.

The SDK does not perform registration, funding, or automatic paid retries.

## Account And API Key

Create a Gradients account and API key outside the SDK, either from the Gradients app or the public account API.

Minimal API bootstrap flow:

```bash
curl -X POST "https://api.gradients.io/account-create" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice"}'

curl -X POST "https://api.gradients.io/auth-with-fingerprint" \
  -H "Content-Type: application/json" \
  -d '{"fingerprint": "<fingerprint>"}'

curl -X POST "https://api.gradients.io/api-key-create" \
  -H "Authorization: Bearer <session_token>"
```

Then configure the SDK:

```bash
export GRADIENTS_API_KEY="..."
export GRADIENTS_SESSION_TOKEN="..."  # Required only for account balance/deposit endpoints.
```

Most SDK calls use `GRADIENTS_API_KEY`. Account functions use `GRADIENTS_SESSION_TOKEN` because the API's balance and deposit endpoints require a session token rather than an API key.

## Balance

Check your account balance before launching paid jobs:

```python
from gradientsio import GradientsClient

client = GradientsClient()  # Uses GRADIENTS_SESSION_TOKEN for client.account.
account = client.account.get_info()

print(account)
```

If you need the deposit address for funding, request your public key:

```python
deposit = client.account.get_public_key()
print(deposit["public_key"])
```

You will need to send TAO to the public key address to get credits.

## Pricing

Check pricing before creating tasks.

For text tasks such as instruct, chat, DPO, or GRPO:

```python
from gradientsio import GradientsClient

client = GradientsClient()

quote = client.tasks.check_text_price(
    model_repo="Qwen/Qwen2.5-7B-Instruct",
    hours_to_complete=1,
)

print(quote.model_dump(exclude_none=True))
```

For image tasks:

```python
quote = client.tasks.check_image_price(
    model_repo="stabilityai/stable-diffusion-xl-base-1.0",
    hours_to_complete=1,
)

print(quote.model_dump(exclude_none=True))
```

You can also fetch the current public price table:

```python
prices = client.tasks.prices()
print(prices)
```

## Tasks

Create a task with `client.train(...)` as shown above. The returned object is a task handle.

Monitor a newly created task:

```python
task = client.train(...)
result = task.wait(poll_interval=600)
print(result.status)
print(result.trained_model_repository)
```

Monitor an existing task:

```python
task = client.tasks.handle("task-id")
details = task.refresh()

print(details.status)
print(details.trained_model_repository)
```

Fetch task details directly:

```python
details = client.tasks.get("task-id")
```

## Scheduler

Use the scheduler for long-running or multi-iteration training where one or more datasets are merged, chunked, and trained across multiple iterations. Each successful iteration can build on the previous merged model. The scheduler currently supports `InstructText`, `Chat`, and `CustomDatasetChat` jobs.

```python
from gradientsio import GradientsClient
from gradientsio import SchedulerDataset

client = GradientsClient()

job = client.scheduler.create_job(
    name="alpaca-iterative-training",
    task_type="InstructText",
    model_repo="Qwen/Qwen2.5-1.5B-Instruct",
    hours_to_complete=1,
    samples_per_training=80000,
    final_test_size=0.1,
    datasets=[
        SchedulerDataset(
            name="yahma/alpaca-cleaned",
            field_instruction="instruction",
            field_input="input",
            field_output="output",
            max_rows=50000,
        ),
        SchedulerDataset(
            name="tatsu-lab/alpaca",
            field_instruction="instruction",
            field_input="input",
            field_output="output",
            max_rows=50000,
        )
    ],
)

job.wait(poll_interval=600)
results = job.results()
print(results.latest_merged_model_repo)
```

Monitor an existing scheduler job:

```python
job = client.scheduler.handle("scheduler-job-id")
details = job.refresh()

print(details.status)

results = job.results()
print(results.latest_merged_model_repo)
```

Or block until the scheduler job reaches a terminal state:

```python
job = client.scheduler.handle("scheduler-job-id")
details = job.wait(poll_interval=600)

print(details.status)
print(job.results().latest_merged_model_repo)
```

## Performance Data

Performance endpoints expose public tournament and weight-projection data. They can be useful to estimate the current tournament winners' emissions and calculate how much alpha would a new winner get on winning. 

```python
weights = client.performance.latest_tournament_weights()
projection = client.performance.weight_projection(percentage_improvement=10.0)
static_projection = client.performance.weight_projection_static()
boss_battle = client.performance.last_boss_battle()
```

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `GRADIENTS_API_KEY` | Gradients API key | None |
| `GRADIENTS_SESSION_TOKEN` | Session token required for account functions, such as balance checks and public key retrieval | None |
