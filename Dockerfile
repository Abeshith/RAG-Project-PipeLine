FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY requirements.txt* ./

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r pyproject.toml

COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
