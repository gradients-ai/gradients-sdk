<img src="assets/banner-scheduler.svg" width="800" alt="Scheduler">

<br>

<img src="assets/section-when-to-use.svg" width="800" alt="When to use the scheduler">

A single `client.train()` call trains on one dataset in one pass. The scheduler is for when that's not enough.

Use the scheduler when you want to train across multiple datasets, run multiple iterations where each builds on the previous result, or both. Each iteration trains, evaluates, and merges the adapter into the base model — then the next iteration starts from that merged model. The result is cumulative improvement over multiple training passes.

This is useful for:
- **Combining diverse data sources** — train on customer support data, then legal documents, then internal FAQs, each as a separate iteration
- **Iterative refinement** — multiple passes over large datasets with different chunks per iteration
- **Progressive training** — start broad, then specialize with increasingly domain-specific data

<br>

---

<img src="assets/section-creating-a-job.svg" width="800" alt="Creating a job">

A scheduler job needs a task type, a base model, and one or more datasets. The scheduler chunks the data, runs training iterations, and merges the results:

```python
from gradientsio import GradientsClient, SchedulerDataset

client = GradientsClient()

job = client.scheduler.create_job(
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
        ),
    ],
)
```

The scheduler splits your data into chunks of `samples_per_training` rows. Each chunk becomes one training iteration. After each iteration, the resulting adapter is merged into the base model, and the next iteration trains on top of that.

`final_test_size` is the proportion of data held out for evaluation (0.1 = 10%).

Supported task types: `InstructText`, `Chat`, `CustomDatasetChat`.

<br>

---

<img src="assets/section-multiple-datasets.svg" width="800" alt="Multiple datasets">

Pass multiple datasets and the scheduler merges and shuffles them before chunking:

```python
job = client.scheduler.create_job(
    task_type="InstructText",
    model_repo="Qwen/Qwen2.5-3B",
    hours_to_complete=1,
    samples_per_training=50000,
    final_test_size=0.1,
    datasets=[
        SchedulerDataset(
            name="yahma/alpaca-cleaned",
            field_instruction="instruction",
            field_input="input",
            field_output="output",
            max_rows=30000,
        ),
        SchedulerDataset(
            name="tatsu-lab/alpaca",
            field_instruction="instruction",
            field_input="input",
            field_output="output",
            max_rows=30000,
        ),
    ],
)
```

`max_rows` limits how many rows to take from each dataset. This is useful when one dataset is much larger than the others and you want balanced representation.

Each dataset can have different column names — set the `field_*` parameters per dataset:

```python
SchedulerDataset(
    name="your-custom-dataset",
    field_instruction="question",
    field_input="context",
    field_output="answer",
)
```

For Chat datasets, use the chat-specific fields:

```python
SchedulerDataset(
    name="your-chat-dataset",
    chat_column="conversations",
    chat_role_field="role",
    chat_content_field="content",
    chat_template="chatml",
)
```

<br>

---

<img src="assets/section-monitoring-jobs.svg" width="800" alt="Monitoring jobs">

Block until a job completes:

```python
job.wait(poll_interval=600)
```

Or check status manually:

```python
details = job.refresh()
print(details.status)    # "pending", "running", "completed", "suspended", "failed"
print(details.is_terminal)
```

List all your scheduler jobs:

```python
jobs = client.scheduler.list()
for j in jobs:
    print(j.id, j.status)
```

Reconnect to an existing job by ID:

```python
job = client.scheduler.handle("your-job-id")
details = job.refresh()
```

Delete a job:

```python
job.delete()
```

<br>

---

<img src="assets/section-results.svg" width="800" alt="Results and merged models">

After a job completes, inspect the results of each iteration:

```python
results = job.results()

for r in results.results:
    print(f"Iteration {r.training_number}")
    print(f"  Status:       {r.status}")
    print(f"  Base model:   {r.base_model_repo}")
    print(f"  Trained:      {r.trained_model_repo}")
    print(f"  Merged:       {r.merged_model_repo}")
    print(f"  Test loss:    {r.test_loss}")
    print(f"  Quality:      {r.quality_score}")
```

The merged model from each iteration becomes the base model for the next. To get the final model:

```python
print(results.latest_merged_model_repo)
```

This is the Hugging Face repo containing the fully merged model after all iterations — ready for inference.

Each `SchedulerTaskResult` contains:

| Field | Description |
|---|---|
| `training_number` | Iteration number (1, 2, 3, ...) |
| `status` | Status of this iteration |
| `base_model_repo` | Base model used for this iteration |
| `trained_model_repo` | LoRA adapter produced by this iteration |
| `merged_model_repo` | Merged model (base + adapter) |
| `test_loss` | Evaluation loss on held-out data |
| `quality_score` | Overall quality score |

<br>

---

<img src="assets/section-job-config.svg" width="800" alt="Job configuration reference">

Full parameter reference for `client.scheduler.create_job()`:

| Parameter | Type | Required | Description |
|---|---|---|---|
| `task_type` | `str` | Yes | `"InstructText"`, `"Chat"`, or `"CustomDatasetChat"` |
| `model_repo` | `str` | Yes | Hugging Face base model |
| `hours_to_complete` | `int` | Yes | Training hours per iteration |
| `samples_per_training` | `int` | Yes | Rows per training iteration |
| `final_test_size` | `float` | Yes | Proportion held out for evaluation (e.g. `0.1`) |
| `datasets` | `list[SchedulerDataset]` | Yes | One or more datasets |
| `name` | `str` | No | Human-readable job name |
| `random_seed` | `int` | No | Seed for reproducible data shuffling |
| `min_days` / `max_days` | `int` | No | Scheduling window in days |
| `min_hours` / `max_hours` | `int` | No | Scheduling window in hours |
| `per_chunk_test_proportion` | `float` | No | Test proportion per chunk (overrides `final_test_size` per iteration) |

`SchedulerDataset` fields:

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Hugging Face dataset repo |
| `max_rows` | No | Limit rows from this dataset |
| `field_instruction`, `field_input`, `field_output` | For Instruct | Column mappings |
| `chat_column`, `chat_role_field`, `chat_content_field`, `chat_template` | For Chat | Chat column mappings |
| `chat_user_reference`, `chat_assistant_reference` | No | Role labels (defaults: `"user"`, `"assistant"`) |

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Inference](inference.md)** — Test your fine-tuned model locally with ModelSampler.
- **[Configuration](configuration.md)** — Training parameters, error handling, and polling behavior.
- **[Datasets](datasets.md)** — Data preparation for each task type.
