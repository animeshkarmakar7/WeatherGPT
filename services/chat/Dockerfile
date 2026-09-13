FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY weathergpt_chat /app/weathergpt_chat

RUN python -m pip install --upgrade pip && python -m pip install .

EXPOSE 8082

CMD ["uvicorn", "weathergpt_chat.api:app", "--host", "0.0.0.0", "--port", "8082"]
