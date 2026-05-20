<img src="assets/banner-configuration.svg" width="800" alt="Configuration">

<br>

This is the reference for every parameter, option, and behavior you can control when using the Gradients SDK.

<br>

<img src="assets/section-train-params.svg" width="800" alt="Training parameters">

The `client.train()` method accepts these core parameters across all task types:

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Hugging Face model repo (e.g. `"Qwen/Qwen2.5-3B"`) |
| `task_type` | `TaskType` | Training mode: `INSTRUCT`, `CHAT`, `DPO`, `GRPO`, `IMAGE` |
| `hours` | `float` | Training duration. Fractional hours are valid (e.g. `0.5`) |
| `dataset` | `str`, `Datasets`, or `dict` | Dataset source — HF repo name, `Datasets.S3(...)`, or raw dict |
| `result_model_name` | `str` (optional) | Custom name for the output model on Hugging Face |
| `yarn_factor` | `int` (optional) | Context length extension. Powers of 2: 2, 4, 8, 16, 32 |

Task-type-specific parameters are covered in [Task Types](task-types.md) and [Datasets](datasets.md).

**Choosing `hours`**

The `hours` parameter controls how long the training job runs, not the wall-clock time you wait. Longer training means more passes over your data. As a rough guide:

| Dataset size | Suggested hours |
|---|---|
| < 1,000 rows | 1 |
| 1,000–10,000 rows | 1–2 |
| 10,000–100,000 rows | 2–4 |
| 100,000+ rows | 4+ |

These are starting points. Gradients optimizes the training configuration automatically — you're setting a time budget, not a hyperparameter.

**`yarn_factor`**

Extends the model's context length beyond its default. Useful when your training examples are longer than the base model's context window. A `yarn_factor` of 4 on a model with 4K context gives you 16K effective context. Only use this if your data actually needs the extra length — it increases memory usage.

<br>

---

<img src="assets/section-task-status.svg" width="800" alt="Task status and lifecycle">

Every training job moves through a sequence of states. You can check the current state at any time:

```python
details = task.refresh()
print(details.status)
```

**Happy path:**

```
PENDING → PREPARING_DATA → LOOKING_FOR_NODES → READY → TRAINING → EVALUATING → SUCCESS
```

| Status | What's happening |
|---|---|
| `PENDING` | Job submitted, queued for processing |
| `PREPARING_DATA` | Dataset being downloaded and validated |
| `LOOKING_FOR_NODES` | Finding available GPUs |
| `DELAYED` | GPUs temporarily unavailable, job is queued |
| `READY` | GPUs allocated, training about to start |
| `TRAINING` | Model is actively training |
| `PREEVALUATION` | Preparing evaluation |
| `EVALUATING` | Running evaluation on held-out test data |
| `SUCCESS` | Training complete. `trained_model_repository` is populated |

**Failure states:**

| Status | What went wrong |
|---|---|
| `PREP_TASK_FAILURE` | Dataset couldn't be prepared — check your field mappings and data format |
| `FAILURE_FINDING_NODES` | No GPUs available after extended wait |
| `FAILURE` | Training failed during execution |

All failure states are terminal. Check the task details for more information:

```python
details = task.refresh()
if details.is_failure:
    print(details.status)
```

**Checking terminal state programmatically:**

```python
details.is_terminal   # True if SUCCESS or any failure state
details.is_success    # True only if SUCCESS
details.is_failure    # True for FAILURE, PREP_TASK_FAILURE, FAILURE_FINDING_NODES
```

<br>

---

<img src="assets/section-error-handling.svg" width="800" alt="Error handling">

The SDK raises typed exceptions you can catch and handle:

```python
from gradientsio import (
    GradientsError,        # base class for all SDK errors
    AuthenticationError,   # 401 — invalid or expired API key
    AuthorizationError,    # 403 — valid key but not permitted
    NotFoundError,         # 404 — task or resource doesn't exist
    ValidationError,       # 400/422 — invalid request (bad params, wrong field names)
    RateLimitError,        # 429 — too many requests
    NetworkError,          # can't reach the API
    TaskFailed,            # task reached a failure state
    TaskTimeout,           # task didn't complete within timeout
)
```

Common patterns:

```python
from gradientsio import ValidationError, TaskFailed

try:
    task = client.train(...)
    result = task.wait()
except ValidationError as e:
    print(f"Bad request: {e}")         # check field names, dataset format
except TaskFailed as e:
    print(f"Training failed: {e}")     # check data, model compatibility
```

`APIError` (parent of the HTTP errors) exposes the status code and request ID for debugging:

```python
from gradientsio import APIError

try:
    task = client.train(...)
except APIError as e:
    print(e.status_code)   # HTTP status
    print(e.request_id)    # for support tickets
```

<br>

---

<img src="assets/section-polling.svg" width="800" alt="Polling and timeouts">

`task.wait()` polls the API until the job reaches a terminal state:

```python
result = task.wait(
    poll_interval=600,        # seconds between checks (default: 600 = 10 min)
    timeout=None,             # max seconds to wait (default: None = forever)
    raise_on_failure=True,    # raise TaskFailed on failure (default: True)
)
```

| Parameter | Default | Description |
|---|---|---|
| `poll_interval` | `600` | Seconds between status checks |
| `timeout` | `None` | Max total wait time. `None` means wait indefinitely |
| `raise_on_failure` | `True` | If `True`, raises `TaskFailed` when the job fails. If `False`, returns the details silently |

If you don't want to block, use `task.refresh()` to check manually:

```python
details = task.refresh()
if details.is_terminal:
    if details.is_success:
        print(details.trained_model_repository)
    else:
        print(f"Failed: {details.status}")
```

**Reconnecting to a task:**

If you lose your session, reconnect by task ID. No state is lost:

```python
task = client.tasks.handle("your-task-id")
result = task.wait()
```

**Retry behavior:**

The SDK automatically retries safe requests (GET, HEAD, OPTIONS) on transient failures, with exponential backoff up to 4 seconds between attempts. Unsafe requests (POST, PUT, DELETE) are never retried — if a `train()` call fails, it failed.

<br>

---

<img src="assets/section-env-vars.svg" width="800" alt="Environment variables">

| Variable | Required | Description |
|---|---|---|
| `GRADIENTS_API_KEY` | Yes | API key for all training and task operations |
| `GRADIENTS_SESSION_TOKEN` | No | Session token — only needed for account balance and deposit operations |
| `RUNPOD_API_KEY` | For RunPod deployment | RunPod API key used by `deploy_runpod()`; read from the environment only |

The API key can also be passed directly to the client:

```python
client = GradientsClient(api_key="your-key")
```

For local inference with `ModelSampler`, you may also need:

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face token for accessing gated models |
| `HUGGING_FACE_HUB_TOKEN` | Alternative HF token variable (either works) |

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Scheduler](scheduler.md)** — Multi-iteration training across multiple datasets.
- **[Inference](inference.md)** — Test your fine-tuned model locally with ModelSampler.
- **[Deployment](deployment.md)** — Deploy a trained LoRA adapter to RunPod with vLLM.
- **[API Reference](api-reference.md)** — Complete class, method, and type reference.
