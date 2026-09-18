FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py schemas.py config.py llm_interpreter.py guardrails.py optimizer.py ./

# .env is NEVER baked in — pass GEMINI_API_KEY[_2.._5] at runtime with -e.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# Prod: 2 workers + bounds survive judge bursts. Local dev stays single
# worker (see README run command). No --reload in prod.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "10", "--limit-concurrency", "50"]
