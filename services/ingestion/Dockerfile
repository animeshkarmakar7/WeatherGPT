FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY weathergpt_ingestion ./weathergpt_ingestion
RUN pip install --no-cache-dir -e .

EXPOSE 8081
CMD ["uvicorn", "weathergpt_ingestion.api:app", "--host", "0.0.0.0", "--port", "8081"]
