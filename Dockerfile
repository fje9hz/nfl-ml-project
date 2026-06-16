FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY models ./models
COPY reports ./reports
COPY web ./web

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "nfl_ml.web:app", "--host", "0.0.0.0", "--port", "8000"]
