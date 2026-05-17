<img src="assets/banner-datasets.svg" width="800" alt="Datasets">

<br>

Every training job needs data. This guide covers where your data can come from, what shape it needs to be in for each task type, and how to get the most out of it.

<br>

<img src="assets/section-sources.svg" width="800" alt="Dataset sources">

Gradients accepts data from three sources:

| Source | How to use it | Best for |
|---|---|---|
| **Hugging Face** | Pass the dataset repo ID as a string | Public datasets, quick experiments |
| **S3 / presigned URL** | Pass a URL to a JSON file | Private data, custom preprocessing |
| **Zip file** | Pass a URL to a zip archive | Image training |

Hugging Face is the simplest — just pass the repo name and Gradients downloads it automatically:

```python
task = client.train(
    dataset="yahma/alpaca-cleaned",
    ...
)
```

For private or custom data, you can host JSON files on S3 (or any URL) and use the `Datasets` helper:

```python
from gradientsio import Datasets

task = client.train(
    dataset=Datasets.S3(
        "https://your-bucket.s3.amazonaws.com/train.json",
        test_data="https://your-bucket.s3.amazonaws.com/test.json",
    ),
    ...
)
```

The rest of this guide covers what your data needs to look like for each task type — and what separates data that works from data that doesn't.

<br>

---

<img src="assets/section-instruct-data.svg" width="800" alt="Instruct datasets">

Instruct datasets are the most straightforward — each row is an instruction and the expected response.

| Field | Parameter | Required | Description |
|---|---|---|---|
| Instruction | `field_instruction` | Yes | The question, prompt, or task |
| Output | `field_output` | Yes | The expected response |
| Input | `field_input` | No | Additional context for the instruction |
| System | `field_system` | No | A system prompt prepended to every example |

Here's what good instruct data looks like:

```json
[
  {
    "instruction": "Summarize this clinical finding.",
    "input": "Patient presents with elevated troponin levels and chest pain radiating to the left arm. ECG shows ST-segment elevation in leads II, III, and aVF.",
    "output": "The patient shows signs of acute inferior myocardial infarction based on elevated troponin and ST elevation in leads II, III, and aVF. Immediate cardiac catheterization is recommended."
  },
  {
    "instruction": "What is the standard dosage of amoxicillin for adults?",
    "input": "",
    "output": "The standard adult dosage is 250–500mg every 8 hours, depending on the severity of infection."
  }
]
```

Notice how the outputs are specific, structured, and consistent in tone. Now compare with data that would train a worse model:

> [!WARNING]
> **Bad example — vague, inconsistent, and lazy outputs:**
> ```json
> {
>   "instruction": "Summarize this clinical finding.",
>   "input": "Patient presents with elevated troponin levels and chest pain...",
>   "output": "The patient may have a heart issue and should see a doctor."
> }
> ```
> The output is too vague ("heart issue"), doesn't reference the actual evidence (troponin, ST elevation), and the recommendation is generic. The model will learn to give non-specific answers.

> [!WARNING]
> **Bad example — instruction does all the work:**
> ```json
> {
>   "instruction": "The patient has elevated troponin and ST elevation in leads II, III, aVF suggesting inferior MI. Summarize this as: acute inferior myocardial infarction requiring immediate catheterization.",
>   "output": "Acute inferior myocardial infarction requiring immediate catheterization."
> }
> ```
> The answer is embedded in the instruction. The model learns to parrot, not reason.

The `input` field is for context the model should reference — a document, a code snippet, a patient history. Leave it empty for standalone questions.

```python
task = client.train(
    model="Qwen/Qwen2.5-3B",
    task_type=TaskType.INSTRUCT,
    hours=2,
    dataset="your-dataset",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)
```

> [!NOTE]
> The parameter names (`field_instruction`, `field_output`, etc.) tell Gradients which columns in your dataset map to which role. If your dataset uses different column names — say `question` and `answer` — just set `field_instruction="question"` and `field_output="answer"`.

<br>

---

<img src="assets/section-chat-data.svg" width="800" alt="Chat datasets">

Chat datasets contain multi-turn conversations. Each row is a full conversation represented as a list of messages.

| Field | Parameter | Required | Description |
|---|---|---|---|
| Conversations | `chat_column` | Yes | Column containing the message list |
| Role key | `chat_role_field` | Yes | Key identifying the speaker (default: `"from"`) |
| Content key | `chat_content_field` | Yes | Key containing the message text (default: `"value"`) |
| Template | `chat_template` | Yes | Format for wrapping turns (default: `"chatml"`) |
| User label | `chat_user_reference` | No | How the user role is labelled (default: `"user"`) |
| Assistant label | `chat_assistant_reference` | No | How the assistant role is labelled (default: `"assistant"`) |

Here's what good chat data looks like:

```json
[
  {
    "conversations": [
      {"from": "user", "value": "I just signed up but I can't find the dashboard."},
      {"from": "assistant", "value": "After signing in, click your profile icon in the top right — the dashboard link is in the dropdown menu. Let me know if you can see it."},
      {"from": "user", "value": "Found it. How do I create my first project?"},
      {"from": "assistant", "value": "On the dashboard, click 'New Project' in the top left. You can choose a template or start from scratch. What are you looking to build?"}
    ]
  }
]
```

The assistant responses are context-aware (referencing what the user just said), ask follow-up questions, and feel like a natural conversation. Compare with:

> [!WARNING]
> **Bad example — ignores context, robotic tone:**
> ```json
> {
>   "conversations": [
>     {"from": "user", "value": "I just signed up but I can't find the dashboard."},
>     {"from": "assistant", "value": "The dashboard can be accessed via the navigation menu."},
>     {"from": "user", "value": "Found it. How do I create my first project?"},
>     {"from": "assistant", "value": "Projects can be created from the dashboard."}
>   ]
> }
> ```
> Every response reads like a FAQ entry — no acknowledgment of what the user said, no follow-up, no personality. The model learns to give canned responses instead of having actual conversations.

The role and content keys must be consistent across all messages. If your data uses `"role"` and `"content"` instead of `"from"` and `"value"`, set the parameters accordingly:

```python
task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.CHAT,
    hours=2,
    dataset="your-chat-dataset",
    chat_column="conversations",
    chat_role_field="role",
    chat_content_field="content",
    chat_template="chatml",
)
```

<br>

---

<img src="assets/section-dpo-data.svg" width="800" alt="DPO datasets">

DPO datasets contain preference pairs — for each prompt, a response that's preferred and one that isn't.

| Field | Parameter | Required | Description |
|---|---|---|---|
| Prompt | `field_prompt` | Yes | The input prompt |
| Chosen | `field_chosen` | Yes | The preferred response |
| Rejected | `field_rejected` | Yes | The non-preferred response |
| System | `field_system` | No | A system prompt |

Here's what good DPO data looks like:

```json
[
  {
    "prompt": "Explain what a DNS server does.",
    "chosen": "A DNS server translates domain names like example.com into IP addresses so your browser knows where to connect. Think of it like a phone book for the internet.",
    "rejected": "DNS stands for Domain Name System. It is a hierarchical and decentralized naming system for computers, services, or other resources connected to the Internet or a private network. It associates various information with domain names assigned to each of the participating entities. Most prominently, it translates more readily memorized domain names to the numerical IP addresses needed for locating and identifying computer services and devices with the underlying network protocols."
  }
]
```

The preference signal is clear: `chosen` is concise and uses an analogy. `rejected` is technically correct but reads like a Wikipedia dump. The model learns the *style* difference.

> [!WARNING]
> **Bad example — no real preference signal:**
> ```json
> {
>   "prompt": "Explain what a DNS server does.",
>   "chosen": "DNS translates domain names to IP addresses.",
>   "rejected": "A DNS server converts domain names into IP addresses."
> }
> ```
> These are effectively the same answer. The model can't learn anything meaningful from pairs where chosen and rejected are too similar. The preference gap needs to be clear and consistent.

> [!WARNING]
> **Bad example — rejected is wrong, not just worse:**
> ```json
> {
>   "prompt": "Explain what a DNS server does.",
>   "chosen": "A DNS server translates domain names to IP addresses.",
>   "rejected": "A DNS server encrypts your internet traffic to prevent hackers from reading it."
> }
> ```
> DPO is for preference, not factual correction. If `rejected` is factually wrong, you're teaching the model "don't say incorrect things" — which is better handled by Instruct training with correct examples. DPO works best when both responses are plausible but one is stylistically better.

A few hundred well-curated pairs often outperform thousands of noisy ones. Focus on consistent, clear preference signals.

```python
task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.DPO,
    hours=2,
    dataset="your-preference-dataset",
    field_prompt="prompt",
    field_chosen="chosen",
    field_rejected="rejected",
)
```

<br>

---

<img src="assets/section-grpo-data.svg" width="800" alt="GRPO datasets">

GRPO datasets are the simplest — you only need prompts. The reward functions you provide separately handle the scoring.

| Field | Parameter | Required | Description |
|---|---|---|---|
| Prompt | `field_prompt` | Yes | The input prompt |
| Extra data | `extra_column` | No | Additional data passed to reward functions |

```json
[
  {"prompt": "Write a Python function that checks if a number is prime."},
  {"prompt": "Convert this SQL query to a pandas operation: SELECT name, COUNT(*) FROM users GROUP BY name"},
  {"prompt": "Explain the difference between TCP and UDP in exactly three sentences."}
]
```

The quality of GRPO training depends almost entirely on your reward functions, not the prompts. But the prompts still matter — they should cover the range of tasks you want the model to handle. If your reward function scores code correctness, make sure your prompts include easy, medium, and hard coding tasks. If it scores format compliance, include prompts that naturally produce varied formats.

> [!WARNING]
> **Bad example — prompts are all the same difficulty:**
> ```json
> [
>   {"prompt": "Write a function that adds two numbers."},
>   {"prompt": "Write a function that subtracts two numbers."},
>   {"prompt": "Write a function that multiplies two numbers."}
> ]
> ```
> The model will learn to handle trivial arithmetic functions and struggle with anything harder. Vary the difficulty and type.

```python
from gradientsio import RewardFunctionReference

task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.GRPO,
    hours=2,
    dataset="your-prompt-dataset",
    field_prompt="prompt",
    reward_functions=[
        RewardFunctionReference(reward_id="your-reward-id", reward_weight=1.0),
    ],
)
```

<br>

---

<img src="assets/section-image-data.svg" width="800" alt="Image datasets">

Image datasets are pairs of images and text captions. You can provide them as a zip file or as individual URLs.

**Zip file format:**

```
my-dataset.zip
├── 0.png
├── 0.txt       ← caption for 0.png
├── 1.jpg
├── 1.txt       ← caption for 1.jpg
├── 2.png
├── 2.txt
└── ...
```

- 10–50 image/caption pairs
- Images: `.png`, `.jpg`, or `.jpeg`
- Captions: `.txt` files with matching base names
- Host the zip at a public or presigned URL

Captions make or break image training. Here's the difference:

> [!IMPORTANT]
> **Good caption:** "A red ceramic coffee mug on a rustic wooden table, soft morning light from the left, shallow depth of field, product photography style"
>
> **Bad caption:** "mug on table"
>
> The model needs specific visual details — color, material, lighting, composition, style — to learn what makes your images distinctive. Generic captions produce generic results.

If you're training a style (illustration, pixel art), describe the style explicitly in every caption. If you're training a subject (a product, a character), use a consistent trigger word like "a photo of [sks] product" across all captions.

```python
task = client.tasks.create_image_zip(
    model_repo="stabilityai/stable-diffusion-xl-base-1.0",
    hours_to_complete=1,
    model_type="sdxl",
    ds="https://your-bucket.s3.amazonaws.com/my-dataset.zip",
)
```

Or using individual URLs:

```python
from gradientsio import ImageTextPair

task = client.tasks.create_image(
    model_repo="stabilityai/stable-diffusion-xl-base-1.0",
    hours_to_complete=1,
    model_type="sdxl",
    image_text_pairs=[
        ImageTextPair(image_url="https://...", text_url="https://..."),
        ImageTextPair(image_url="https://...", text_url="https://..."),
    ],
)
```

<br>

---

<img src="assets/section-custom-data.svg" width="800" alt="Custom datasets via S3">

For private or preprocessed data, host your JSON files on S3 (or any URL that returns JSON) and use the `Datasets` helper. This works with Instruct and Chat task types.

```python
from gradientsio import Datasets

task = client.train(
    model="Qwen/Qwen2.5-3B",
    task_type=TaskType.INSTRUCT,
    hours=2,
    dataset=Datasets.S3(
        "https://your-bucket.s3.amazonaws.com/train.json",
        test_data="https://your-bucket.s3.amazonaws.com/test.json",
    ),
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)
```

The `test_data` parameter is optional. If provided, Gradients uses it as a held-out evaluation set. If omitted, Gradients splits the training data automatically.

Your JSON files should be arrays of objects with the same field structure as the corresponding task type (see sections above).

<br>

---

<img src="assets/section-tips.svg" width="800" alt="Preparing good training data">

A few things that make a real difference:

- **Consistency matters more than volume.** 500 high-quality examples with a consistent format and style will outperform 50,000 noisy ones. If your data has mixed formats, inconsistent tone, or contradictory examples, the model will learn that inconsistency.
- **Match the format you want at inference time.** If you want the model to answer in bullet points, your training examples should use bullet points. If you want two-sentence answers, don't train on paragraphs. The model reproduces what it sees.
- **Use a held-out test set.** Either provide one via `test_data` or let Gradients split automatically. Without a test set, you're flying blind — you won't know if the model actually learned or just memorized.
- **Column names are flexible.** Your dataset doesn't need to use `instruction`/`input`/`output` as column names. Use whatever makes sense — just set the `field_*` parameters to match.
- **Start small.** Run a short training job (1 hour) on a small model first to validate your data format and field mappings before committing to a longer run on a larger model.
- **Deduplicate.** Repeated examples don't help — they just make the model overfit to those specific responses. Remove exact and near-duplicates before training.

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Task Types](task-types.md)** — Detailed guide for each training mode, with scenarios and what happens under the hood.
- **[Configuration](configuration.md)** — Every parameter you can control, plus error handling and failure states.
- **[Getting Started](getting-started.md)** — End-to-end walkthrough from install to testing a trained model.
