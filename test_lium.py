# test_basilica_deploy.py
# ruff: noqa: I001
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradientsio  # noqa: E402

base_model = "Qwen/Qwen2.5-Coder-32B-Instruct"
lora = "gradients-io-tournaments/organic_662326e7-4c35-47ec-876e-f5d4b9089878"

if not os.getenv("BASILICA_API_KEY"):
    raise RuntimeError("Set BASILICA_API_KEY before running this script.")

deployment = gradientsio.deploy_basilica(
    base_model=base_model,
    lora=lora,
    wait=False,
    max_model_len=4096,
    gpu_memory_utilization=0.88,
    dtype="half",
    enforce_eager=True,
)

print("Deployment ID:", deployment.id)
print("Server URL:", deployment.server_url)
print("Status:", deployment.details.status)

try:
    if not deployment.server_url:
        raise RuntimeError("Basilica deployment started, but no server URL was returned. Check the Basilica dashboard.")

    print("Waiting for vLLM to finish provisioning...")
    deployment.wait_ready(poll_interval=20)

    sampler = deployment.sampler(timeout=120)

    prompts = [
        "Write a Python function that checks whether a number is prime.\n\nAnswer:",
        "Explain the difference between a list and a tuple in Python.\n\nAnswer:",
        "Create a simple FastAPI endpoint that returns hello world.\n\nAnswer:",
    ]

    outputs = sampler.generate(
        prompts,
        max_tokens=256,
        temperature=0.2,
        top_p=0.95,
    )

    for i, output in enumerate(outputs, start=1):
        print(f"\n--- Output {i} ---")
        print(output)
finally:
    print(f"\nDeleting Basilica deployment {deployment.id}...")
    deployment.delete()
    print("Deleted Basilica deployment.")