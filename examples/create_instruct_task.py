from gradients import GradientsClient


client = GradientsClient()

price = client.tasks.check_text_price(
    model_repo="Qwen/Qwen2.5-7B-Instruct",
    hours_to_complete=1,
)
print(f"Price quote: {price.model_dump(exclude_none=True)}")

task = client.tasks.create_instruct(
    ds_repo="yahma/alpaca-cleaned",
    model_repo="Qwen/Qwen2.5-7B-Instruct",
    file_format="hf",
    hours_to_complete=1,
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)

print(f"Created task: {task.task_id}")
result = task.wait(poll_interval=600)
print(f"Trained model: {result.trained_model_repository}")
