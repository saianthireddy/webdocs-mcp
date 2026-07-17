FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY pyproject.toml .

ENV PYTHONPATH=/app/src \
    WEBDOCS_DB_PATH=/data/webdocs.duckdb

VOLUME /data
EXPOSE 9111

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9111/health')"

CMD ["uvicorn", "webdocs.api:app", "--host", "0.0.0.0", "--port", "9111"]
