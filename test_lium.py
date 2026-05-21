# test_lium_deploy.py
# ruff: noqa: I001
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradientsio  # noqa: E402

base_model = "Qwen/Qwen2.5-0.5B-Instruct"

if not os.getenv("LIUM_API_KEY"):
    raise RuntimeError("Set LIUM_API_KEY before running this script.")

deployment = gradientsio.deploy_lium(
    base_model=base_model,
    gpu_count=1,  # override if you want more GPUs
    gpu_type="H100",
    wait=True,
    max_model_len=2048,
    gpu_memory_utilization=0.88,
    dtype="half",
    enforce_eager=True,
)

print("Deployment ID:", deployment.id)
print("Server URL:", deployment.server_url)
print("Status:", deployment.status())

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