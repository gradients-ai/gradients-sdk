from gradients import GradientsClient


client = GradientsClient()
task = client.tasks.handle("00000000-0000-0000-0000-000000000000")

details = task.refresh()
print(f"Task status: {details.status}")

if details.is_success:
    print(f"Trained model: {details.trained_model_repository}")
