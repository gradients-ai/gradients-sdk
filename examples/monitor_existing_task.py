from gradientsio import GradientsClient


client = GradientsClient()
task = client.tasks.handle("e96a2dec-03f1-4143-946f-009724bd807e")

details = task.refresh()
print(f"Task details: {details.model_dump(exclude_none=True)}")
print(f"Task status: {details.status}")

if details.is_success:
    print(f"Trained model: {details.trained_model_repository}")
