FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY README.md LICENSE requirements.txt 1.html ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/1.html', timeout=2).read(1)"

CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0"]
