FROM python:3.10

WORKDIR /app

COPY ./rest_service/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./rest_service /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
