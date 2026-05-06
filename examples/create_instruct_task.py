from gradientsio import GradientsClient
from gradientsio import TaskType


client = GradientsClient()

price = client.tasks.check_text_price(
    model_repo="Qwen/Qwen2.5-7B-Instruct",
    hours_to_complete=1,
)
print(f"Price quote: {price.model_dump(exclude_none=True)}")

task = client.train(
    model="Qwen/Qwen2.5-7B-Instruct",
    task_type=TaskType.INSTRUCT,
    hours=1,
    dataset="yahma/alpaca-cleaned",
    field_instruction="instruction",
    field_input="input",
    field_output="output",
)

print(f"Created task: {task.task_id}")
result = task.wait(poll_interval=600)
print(f"Trained model: {result.trained_model_repository}")
