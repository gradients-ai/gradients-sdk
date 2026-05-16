<img src="assets/banner-task-types.svg" width="800" alt="Task Types">

<br>

<img src="assets/section-which-type.svg" width="800" alt="Which type do I use?">

Match your data to a training mode.

| I have... | Use | Task type |
|---|---|---|
| Question/answer pairs, or instruction/response data | **[Instruct](#instruct)** | `TaskType.INSTRUCT` |
| Multi-turn conversations | **[Chat](#chat)** | `TaskType.CHAT` |
| Pairs of good and bad responses to the same prompt | **[DPO](#dpo)** | `TaskType.DPO` |
| Prompts and a way to score outputs programmatically | **[GRPO](#grpo)** | `TaskType.GRPO` |
| 10–50 images with captions | **[Image](#image)** | `TaskType.IMAGE` |

<br>

---

<a id="instruct"></a>
<img src="assets/task-instruct.svg" width="800" alt="Instruct">

Supervised fine-tuning on instruction/response pairs.

Instruct training teaches a model to follow instructions by showing it examples of questions and correct answers. This is the most common and straightforward way to fine-tune a model.

Use Instruct when you have structured data where each row is a task and a desired response. Domain-specific QA, customer support responses, code generation, document summarization — anything where you can express the training data as "given this input, produce this output."

Dataset fields:

| Field | Required | Description |
|---|---|---|
| `field_instruction` | Yes | The question, prompt, or task |
| `field_output` | Yes | The expected response |
| `field_input` | No | Additional context (e.g. a document to reference) |
| `field_system` | No | A system prompt |

Example row:

| instruction | input | output |
|---|---|---|
| Summarize this clinical finding. | Patient presents with elevated troponin levels... | The patient shows signs of acute myocardial injury... |

For full dataset preparation details, see [Datasets](datasets.md).

With your data ready, training is a single call:

```python
from gradientsio import GradientsClient, TaskType

client = GradientsClient()

task = client.train(
    model="Qwen/Qwen2.5-3B",
    task_type=TaskType.INSTRUCT,
    hours=2,
    dataset="your-hf-dataset",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)

result = task.wait()
print(result.trained_model_repository)
```

What happens behind the scenes: Instruct training runs supervised fine-tuning (SFT) using LoRA adapters. The model learns to map instructions to outputs by minimizing the difference between its predictions and your training examples. LoRA keeps the process efficient — rather than updating every parameter in the model, it trains a small set of adapter weights that modify the model's behavior. The result is a lightweight adapter that can be loaded on top of the base model for inference.

<br>

---

<a id="chat"></a>
<img src="assets/task-chat.svg" width="800" alt="Chat">

Fine-tune on multi-turn conversations.

Chat training teaches a model to handle back-and-forth dialogue. Instead of single instruction/response pairs, your data is full conversations with multiple turns between a user and an assistant.

Use Chat when your data is naturally conversational — chatbot logs, support transcripts, tutoring sessions, or any scenario where context builds across multiple messages. The model learns not just how to respond, but how to track context across a conversation.

Dataset fields:

| Field | Required | Description |
|---|---|---|
| `chat_column` | Yes | Column containing the conversation (list of messages) |
| `chat_role_field` | Yes | Key for the speaker role in each message (default: `"from"`) |
| `chat_content_field` | Yes | Key for the message text (default: `"value"`) |
| `chat_template` | Yes | Conversation format (default: `"chatml"`) |
| `chat_user_reference` | No | How the user role is labelled (default: `"user"`) |
| `chat_assistant_reference` | No | How the assistant role is labelled (default: `"assistant"`) |

Example row:

```json
{
  "conversations": [
    {"from": "user", "value": "What causes type 2 diabetes?"},
    {"from": "assistant", "value": "Type 2 diabetes develops when the body becomes resistant to insulin..."},
    {"from": "user", "value": "How is it diagnosed?"},
    {"from": "assistant", "value": "Diagnosis typically involves fasting blood glucose tests..."}
  ]
}
```

For full dataset preparation details, see [Datasets](datasets.md).

With your conversations formatted, training looks like this:

```python
task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.CHAT,
    hours=2,
    dataset="your-chat-dataset",
    chat_column="conversations",
    chat_role_field="from",
    chat_content_field="value",
    chat_template="chatml",
)

result = task.wait()
print(result.trained_model_repository)
```

What happens behind the scenes: Chat training is supervised fine-tuning applied to conversation-formatted data. The model is trained using a chat template (like ChatML) that wraps each turn with role markers, so the model learns when it's the assistant's turn to speak and how to condition its response on the full conversation history. Like Instruct, it produces a LoRA adapter.

<br>

---

<a id="dpo"></a>
<img src="assets/task-dpo.svg" width="800" alt="DPO">

Preference-based training from chosen vs rejected responses.

DPO (Direct Preference Optimization) trains a model to prefer better responses over worse ones. Instead of showing the model "here's the right answer," you show it two answers to the same prompt and tell it which one is better.

Use DPO when you want to steer a model's behavior — improving tone, reducing harmful outputs, aligning with a house style, or teaching it to prefer concise answers over verbose ones. It's particularly effective when you already have a model that's roughly capable but needs to be refined in how it responds. DPO is the standard approach for alignment and preference tuning.

Dataset fields:

| Field | Required | Description |
|---|---|---|
| `field_prompt` | Yes | The input prompt |
| `field_chosen` | Yes | The preferred response |
| `field_rejected` | Yes | The non-preferred response |
| `field_system` | No | A system prompt |

Example row:

| prompt | chosen | rejected |
|---|---|---|
| Explain quantum entanglement simply. | When two particles are entangled, measuring one instantly tells you about the other... | Quantum entanglement is a phenomenon described by the EPR paradox whereby... |

For full dataset preparation details, see [Datasets](datasets.md).

With your preference pairs ready, training is one call:

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

result = task.wait()
print(result.trained_model_repository)
```

What happens behind the scenes: DPO skips the reward model step used in traditional RLHF. Instead, it directly optimizes the model's policy to assign higher probability to chosen responses and lower probability to rejected ones. This requires more GPU memory than Instruct training (roughly 3x) because the algorithm needs to compare the model's current behavior against a reference. The result is a LoRA adapter that shifts the model's preferences without changing its core capabilities.

> [!IMPORTANT]
> DPO uses approximately 3x the GPU memory of Instruct training. This is reflected in pricing for larger models.

<br>

---

<a id="grpo"></a>
<img src="assets/task-grpo.svg" width="800" alt="GRPO">

Reward-driven training using custom scoring functions.

GRPO (Group Relative Policy Optimization) trains a model using reward functions that score its outputs programmatically. Instead of providing correct answers or preference pairs, you define what "good" looks like as code, and the model learns to maximize that score.

Use GRPO when the quality of an output can be measured automatically — code correctness (does it compile?), format compliance (does it follow a schema?), length constraints, reasoning quality, or any custom criteria. GRPO is especially useful when you can't easily write out ideal answers but you can write a function that scores them.

Your dataset needs a prompt column, and you provide reward functions separately:

| Field | Required | Description |
|---|---|---|
| `field_prompt` | Yes | The input prompt |
| `reward_functions` | Yes | List of reward function references with weights |
| `extra_column` | No | Additional data passed to reward functions |

```python
from gradientsio import RewardFunctionReference

rewards = [
    RewardFunctionReference(reward_id="your-reward-func-id", reward_weight=1.0),
]
```

For full dataset preparation details, see [Datasets](datasets.md).

With your prompts and reward functions defined, training looks like this:

```python
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

result = task.wait()
print(result.trained_model_repository)
```

What happens behind the scenes: GRPO generates multiple completions for each prompt, scores them with your reward functions, and uses the relative scores within each group to update the model. Higher-scoring completions get reinforced, lower-scoring ones get suppressed. It uses roughly 2x the GPU memory of Instruct training because it needs to generate and evaluate multiple completions per step. The result is a LoRA adapter that steers the model toward outputs that score well on your criteria.

<br>

---

<a id="image"></a>
<img src="assets/task-image.svg" width="800" alt="Image">

LoRA fine-tuning for image generation models.

Image training teaches a diffusion model to generate images in a specific style or of a specific subject. You provide a small set of images with captions, and the model learns to reproduce that visual concept.

Use Image training when you want a model that generates images of a particular style (illustration, pixel art, architectural renders), a specific subject (a product, a character, a brand aesthetic), or a visual concept that doesn't exist in the base model's training data. You only need 10–50 images.

Image training expects a zip file containing image/caption pairs with matching filenames:

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
- Captions: `.txt` files with the same base name
- Provide a public or presigned URL to the zip

Alternatively, you can pass image/text pairs directly as URLs. See [Datasets](datasets.md) for details.

Training with a zip file:

```python
task = client.tasks.create_image_zip(
    model_repo="stabilityai/stable-diffusion-xl-base-1.0",
    hours_to_complete=1,
    model_type="sdxl",
    ds="https://your-bucket.s3.amazonaws.com/my-dataset.zip",
)

result = task.wait()
print(result.trained_model_repository)
```

Or using individual image/text pair URLs:

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

result = task.wait()
print(result.trained_model_repository)
```

> [!TIP]
> Supported image models: `sdxl` (Stable Diffusion XL) and `flux` (Flux variants).

What happens behind the scenes: Image training produces a LoRA adapter for the diffusion model. For SDXL, images are trained with 10 repeats for style concepts and 8 for subject concepts, giving the model enough exposure to learn the visual pattern from a small dataset. Flux models use a different training curve with single repeats. The adapter modifies the model's attention layers to associate your captions with the visual features in your images. All image training runs on a single A100 GPU.

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Datasets](datasets.md)** — How to prepare, format, and upload your data for each task type.
- **[Configuration](configuration.md)** — Every parameter you can control, plus error handling and failure states.
- **[Getting Started](getting-started.md)** — End-to-end walkthrough from install to testing a trained model.
