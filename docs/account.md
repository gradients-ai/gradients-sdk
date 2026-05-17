<img src="assets/banner-account.svg" width="800" alt="Account">

<br>

<img src="assets/section-account.svg" width="800" alt="Account and billing">

Create your account at [gradients.io](https://www.gradients.io/) — one-click sign-up. From your dashboard, generate an API key.

Training jobs are funded with TAO. To get your deposit address and check your balance:

```python
# get TAO deposit address
deposit = client.account.get_public_key()
print(deposit["public_key"])

# check balance
account = client.account.get_info()
print(account)
```

Transfer TAO to the deposit address to fund your account. Balance updates are reflected immediately.

> [!NOTE]
> Account operations require `GRADIENTS_SESSION_TOKEN` in addition to your API key.

<br>

---

<img src="assets/section-pricing.svg" width="800" alt="Pricing">

Pay per hour of training. The rate depends on model size.

| Model size | Hourly rate | | |
|---|---|---|---|
| Up to 1B parameters | $10 / hr | 40B+ parameters | $50 / hr |
| Up to 7B parameters | $15 / hr | Image models | $5 / hr |
| Up to 40B parameters | $25 / hr | | |

Check the exact cost of a specific job before running it:

```python
# text models
quote = client.tasks.check_text_price(
    model_repo="Qwen/Qwen2.5-7B-Instruct",
    hours_to_complete=2,
)
print(quote.total_price)

# image models
quote = client.tasks.check_image_price(
    model_repo="stabilityai/stable-diffusion-xl-base-1.0",
    hours_to_complete=1,
)
print(quote.total_price)

# full price table
prices = client.tasks.prices()
```

<br>

---

<img src="assets/section-what-to-read-next.svg" width="800" alt="What to read next">

- **[Getting Started](getting-started.md)** — End-to-end walkthrough from install to testing a trained model.
- **[Task Types](task-types.md)** — Explore the five training modes and find the right one for your data.
