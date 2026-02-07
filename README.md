# Reef Server

FastAPI server for the Reef iOS app. Provides PDF reconstruction and text embedding services.

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/ai/embed` | POST | Text embeddings (MiniLM-L6-v2) |
| `/ai/annotate` | POST | PDF layout annotation (Surya) |
| `/ai/group-problems` | POST | Group annotations into problems (Gemini) |
| `/ai/reconstruct` | POST | Full PDF reconstruction pipeline |

## PDF Reconstruction Pipeline

`PDF pages -> Surya layout detection -> Gemini problem grouping -> OpenAI structured extraction -> LaTeX -> tectonic compilation -> merged PDF`

## Local Development

```bash
# Install dependencies with uv
uv sync --all-extras

# Copy environment variables
cp .env.example .env
# Edit .env with your OPENROUTER_API_KEY

# Run server
uv run uvicorn api.index:app --host 0.0.0.0 --port 8080
```

## Deployment

Deployed on Railway via Docker.
