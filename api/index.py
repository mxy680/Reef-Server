"""
Reef Server - FastAPI application for proxying Gemini API calls.

Provides:
- Gemini API proxy (keeps API key server-side)
- Mock mode for testing
- Error simulation for testing error handling
- Latency injection for timeout testing
"""

from fastapi import FastAPI, Request, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import os
from dotenv import load_dotenv

# Import lib modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gemini_proxy import GeminiProxy
from lib.mock_responses import get_mock_response
from lib.simulator import simulate_error, simulate_delay
from lib.logger import RequestLogger

load_dotenv()

app = FastAPI(
    title="Reef Server",
    description="API proxy for Reef iOS app",
    version="1.0.0"
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
gemini_proxy = GeminiProxy(api_key=os.getenv("GEMINI_API_KEY"))
request_logger = RequestLogger()


# Request/Response Models
class GenerateRequest(BaseModel):
    """Request body for text generation."""
    prompt: str
    json_output: bool = False
    response_schema: Optional[dict] = None
    generation_config: Optional[dict] = None


class VisionRequest(BaseModel):
    """Request body for multimodal (vision) generation."""
    prompt: str
    images: list[dict]  # [{"data": "base64...", "mime_type": "image/png"}]
    json_output: bool = False
    response_schema: Optional[dict] = None
    generation_config: Optional[dict] = None


class GenerateResponse(BaseModel):
    """Response from generation endpoints."""
    text: str
    model: str = "gemini-2.5-flash"
    mode: str = "prod"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    code: str


# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "reef-server",
        "version": "1.0.0"
    }


@app.get("/logs")
async def get_logs(limit: int = Query(default=100, le=1000)):
    """
    Get recent request logs (development only).

    Returns the last N logged requests with timing and status info.
    """
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(status_code=403, detail="Logs not available in production")

    return {"logs": request_logger.get_logs(limit)}


@app.post("/gemini/generate", response_model=GenerateResponse)
async def generate_content(
    request_body: GenerateRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
    delay: Optional[int] = Query(default=None, ge=0, le=30000),
    error: Optional[str] = Query(default=None, pattern="^(rate_limit|timeout|500)$"),
    x_mock_scenario: Optional[str] = Header(default=None, alias="X-Mock-Scenario"),
):
    """
    Proxy text generation to Gemini API.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real API
    - delay: Add latency in milliseconds (0-30000)
    - error: Simulate error ("rate_limit", "timeout", "500")

    Headers:
    - X-Mock-Scenario: Select specific mock response scenario
    """
    # Log request
    log_id = request_logger.log_request(
        endpoint="/gemini/generate",
        mode=mode,
        prompt_preview=request_body.prompt[:100] if request_body.prompt else ""
    )

    try:
        # Simulate delay if requested
        if delay:
            await simulate_delay(delay)

        # Simulate error if requested
        if error:
            simulate_error(error)

        # Mock mode
        if mode == "mock":
            text = get_mock_response(
                endpoint="generate",
                scenario=x_mock_scenario,
                prompt=request_body.prompt
            )
            request_logger.log_response(log_id, success=True, mode="mock")
            return GenerateResponse(text=text, mode="mock")

        # Production mode - proxy to Gemini
        text = await gemini_proxy.generate_content(
            prompt=request_body.prompt,
            json_output=request_body.json_output,
            schema=request_body.response_schema,
            generation_config=request_body.generation_config
        )

        request_logger.log_response(log_id, success=True, mode="prod")
        return GenerateResponse(text=text, mode="prod")

    except HTTPException:
        raise
    except Exception as e:
        request_logger.log_response(log_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gemini/vision", response_model=GenerateResponse)
async def generate_vision_content(
    request_body: VisionRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
    delay: Optional[int] = Query(default=None, ge=0, le=30000),
    error: Optional[str] = Query(default=None, pattern="^(rate_limit|timeout|500)$"),
    x_mock_scenario: Optional[str] = Header(default=None, alias="X-Mock-Scenario"),
):
    """
    Proxy multimodal (vision) generation to Gemini API.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real API
    - delay: Add latency in milliseconds (0-30000)
    - error: Simulate error ("rate_limit", "timeout", "500")

    Headers:
    - X-Mock-Scenario: Select specific mock response scenario
    """
    # Log request
    log_id = request_logger.log_request(
        endpoint="/gemini/vision",
        mode=mode,
        prompt_preview=request_body.prompt[:100] if request_body.prompt else "",
        image_count=len(request_body.images)
    )

    try:
        # Simulate delay if requested
        if delay:
            await simulate_delay(delay)

        # Simulate error if requested
        if error:
            simulate_error(error)

        # Mock mode
        if mode == "mock":
            text = get_mock_response(
                endpoint="vision",
                scenario=x_mock_scenario,
                prompt=request_body.prompt
            )
            request_logger.log_response(log_id, success=True, mode="mock")
            return GenerateResponse(text=text, mode="mock")

        # Production mode - proxy to Gemini
        text = await gemini_proxy.generate_content_with_images(
            prompt=request_body.prompt,
            images=request_body.images,
            json_output=request_body.json_output,
            schema=request_body.response_schema,
            generation_config=request_body.generation_config
        )

        request_logger.log_response(log_id, success=True, mode="prod")
        return GenerateResponse(text=text, mode="prod")

    except HTTPException:
        raise
    except Exception as e:
        request_logger.log_response(log_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
