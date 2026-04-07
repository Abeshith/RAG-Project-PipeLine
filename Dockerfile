FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# Health check for HF Spaces - this endpoint returns immediately
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--startup-timeout", "3600"]
