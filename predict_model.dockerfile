# Base image
FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

COPY pyproject.toml pyproject.toml
COPY MLOpsProject/ MLOpsProject/
COPY data/ data/

ENTRYPOINT ["python", "-u", "MLOpsProject/predict_model.py"]