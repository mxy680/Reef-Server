# Reef Server

FastAPI server for Reef iOS app. Proxies Gemini API calls and provides testing infrastructure.

## Features

- **Gemini API Proxy**: Secure proxy for Gemini API calls (keeps API key server-side)
- **Mock Mode**: Test without hitting real API
- **Error Simulation**: Test error handling with simulated failures
- **Latency Injection**: Test timeout handling

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/gemini/generate` | POST | Proxy text generation |
| `/gemini/vision` | POST | Proxy multimodal requests |
| `/health` | GET | Health check |
| `/logs` | GET | View request logs (dev only) |

## Query Parameters

- `mode=mock` or `mode=prod` - Select mock or production mode
- `delay=2000` - Add latency in milliseconds
- `error=rate_limit` - Simulate 429 error
- `error=timeout` - Simulate timeout
- `error=500` - Simulate server error

## Headers

- `X-Mock-Scenario: <scenario>` - Select specific mock response

## Local Development

```bash
# Install dependencies with uv
uv sync --all-extras

# Copy environment variables
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Run server
uv run uvicorn api.index:app --reload
```

## Testing

```bash
# Run tests
uv run pytest

# Test mock mode
curl -X POST http://localhost:8000/gemini/generate?mode=mock \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'

# Test with delay
curl -X POST http://localhost:8000/gemini/generate?mode=mock&delay=2000 \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

## Deployment

Deployed on Vercel. Set `GEMINI_API_KEY` environment variable in Vercel dashboard.

```bash
vercel --prod
```
