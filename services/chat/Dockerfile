FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY weathergpt_chat ./weathergpt_chat
RUN pip install --no-cache-dir -e .

EXPOSE 8082
CMD ["uvicorn", "weathergpt_chat.api:app", "--host", "0.0.0.0", "--port", "8082"]
