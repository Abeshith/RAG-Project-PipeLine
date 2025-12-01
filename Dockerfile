FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt* ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
