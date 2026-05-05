from gradients import GradientsClient
from gradients import SchedulerDataset


client = GradientsClient()

job = client.scheduler.create_job(
    name="alpaca-iterative-training",
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
        )
    ],
)

print(f"Created scheduler job: {job.job_id}")
details = job.wait(poll_interval=600)
print(f"Scheduler status: {details.status}")
print(f"Latest merged model: {job.results().latest_merged_model_repo}")
