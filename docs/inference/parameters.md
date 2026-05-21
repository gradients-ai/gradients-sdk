# Inference Parameters

`RemoteVLLMSampler.generate()` and `RemoteVLLMSampler.chat()` expose the common OpenAI-compatible request parameters used by vLLM servers. `ModelSampler` passes these through when using its default local vLLM backend.

Example:

```python
answers = sampler.generate(
    ["What is DNA?"],
    max_tokens=128,
    temperature=0.2,
    top_p=0.95,
    stop=["\n\n"],
    seed=7,
)
```

For parameters not listed here, pass provider-specific extras as keyword arguments. They are included in the JSON request body.

## Shared Parameters

| Parameter | Default | Applies to | Description |
|---|---|---|---|
| `config` | `GenerationConfig()` | completions, chat | SDK config object. `max_new_tokens`, `do_sample`, and `repetition_penalty` are used as defaults for server requests. |
| `max_tokens` | `config.max_new_tokens` (`96`) | completions, chat | Maximum number of output tokens. |
| `temperature` | `0` when `do_sample=False`, otherwise `0.7` | completions, chat | Sampling temperature. `0` is deterministic. |
| `top_p` | `1.0` | completions, chat | Nucleus sampling cutoff. |
| `n` | `1` | completions, chat | Number of completions to generate. |
| `stream` | `False` | completions, chat | Streaming flag. The SDK currently raises if set to `True`; use direct HTTP for streaming. |
| `stop` | `None` | completions, chat | Stop sequence or list of stop sequences. |
| `presence_penalty` | `0.0` | completions, chat | Penalizes tokens that already appeared. |
| `frequency_penalty` | `0.0` | completions, chat | Penalizes tokens based on frequency. |
| `repetition_penalty` | `config.repetition_penalty` (`1.12`) | completions, chat | vLLM repetition penalty. |
| `logit_bias` | `None` | completions, chat | Bias specific token IDs. |
| `user` | `None` | completions, chat | Optional end-user identifier passed through to the server. |
| `seed` | `None` | completions, chat | Random seed for reproducible sampling when supported. |
| `top_k` | `None` | completions, chat | vLLM top-k sampling. |
| `min_p` | `None` | completions, chat | vLLM minimum probability sampling. |
| `min_tokens` | `None` | completions, chat | Minimum number of tokens to generate when supported. |
| `ignore_eos` | `False` | completions, chat | Continue generation after EOS when supported. |
| `skip_special_tokens` | `True` | completions, chat | Omit special tokens from returned text. |
| `spaces_between_special_tokens` | `True` | completions, chat | Preserve spaces between special tokens. |

## Completion-Only Parameters

| Parameter | Default | Description |
|---|---|---|
| `best_of` | `None` | Generate multiple candidates server-side and return the best when supported. |
| `logprobs` | `None` | Return token log probabilities. |
| `prompt_logprobs` | `None` | Return prompt token log probabilities when supported by vLLM. |
| `echo` | `False` | Include the prompt in the returned completion text. |
| `suffix` | `None` | Text appended after the generated completion when supported. |

## Chat-Only Parameters

| Parameter | Default | Description |
|---|---|---|
| `response_format` | `None` | OpenAI-style response format, such as JSON object mode when supported. |
| `tools` | `None` | Tool definitions for tool-calling models. |
| `tool_choice` | `None` | Tool choice policy or selected tool. |

## Direct HTTP

The same fields can be sent directly to the server:

```bash
curl "$SERVER_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-served-model-name",
    "prompt": "What is DNA?",
    "max_tokens": 128,
    "temperature": 0.2,
    "top_p": 0.95
  }'
```
