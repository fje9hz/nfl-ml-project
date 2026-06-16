# Deployment

The app is ready to deploy as a Python web service. The simplest path is Render.

## Render

1. Open Render and create a new web service from the GitHub repo.
2. Select this repository: `fje9hz/nfl-ml-project`.
3. Render should detect `render.yaml` automatically.
4. If entering settings manually, use:

```text
Build Command: pip install -r requirements.txt
Start Command: python3 -m uvicorn nfl_ml.web:app --host 0.0.0.0 --port $PORT
Health Check Path: /api/health
Python Runtime: python-3.11.11
```

After the first deploy finishes, Render gives you a public `onrender.com` URL.

## Local Production Check

```bash
python3 -m uvicorn nfl_ml.web:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Docker

Build and run locally:

```bash
docker build -t nfl-win-probability .
docker run --rm -p 8000:8000 nfl-win-probability
```

The container runs:

```text
python -m uvicorn nfl_ml.web:app --host 0.0.0.0 --port 8000
```

You can deploy this image to Docker-friendly hosts such as Google Cloud Run,
Fly.io, Railway, DigitalOcean App Platform, AWS App Runner, or Hugging Face
Spaces.
