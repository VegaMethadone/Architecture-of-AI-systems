from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import subprocess
import docker

app = FastAPI()

docker_client = docker.from_env()


@app.post('/deploy_hook')
def deploy_hook():
    image_tag = "radmirkus/architecture_ai_rest_service"

    containers = docker_client.containers.list(filters={"ancestor": image_tag})
    for container in containers:
        container.restart()
        print(f"container {container.id} restarted")

    return { 'status': 'ok' }

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
