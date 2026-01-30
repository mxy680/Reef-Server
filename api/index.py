"""
Reef Server - FastAPI application for proxying AI API calls.

Provides:
- Multi-provider AI proxy (OpenAI, Anthropic, Google)
- AI endpoints for feedback, quiz generation, and chat
- Mock mode for testing
- Error simulation for testing error handling
- Latency injection for timeout testing
"""

from fastapi import FastAPI, Request, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import os
import json
from dotenv import load_dotenv

# Import lib modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.gemini_proxy import GeminiProxy
from lib.mock_responses import get_mock_response
from lib.simulator import simulate_error, simulate_delay
from lib.logger import RequestLogger
from lib.providers import ProviderRouter, GenerationConfig
from lib.models import (
    FeedbackRequest, FeedbackResponse,
    QuizRequest, QuizResponse, QuizQuestion, QuizOption, QUIZ_RESPONSE_SCHEMA,
    ChatRequest, ChatResponse, SourceReference,
)
from lib.prompt_templates import (
    build_feedback_prompt,
    build_quiz_prompt,
    build_chat_single_prompt,
)

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
provider_router = ProviderRouter()


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


# ============================================================================
# AI Endpoints - Multi-provider support
# ============================================================================

@app.post("/ai/feedback", response_model=FeedbackResponse)
async def ai_feedback(
    request_body: FeedbackRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
    delay: Optional[int] = Query(default=None, ge=0, le=30000),
    error: Optional[str] = Query(default=None, pattern="^(rate_limit|timeout|500)$"),
    x_mock_scenario: Optional[str] = Header(default=None, alias="X-Mock-Scenario"),
):
    """
    Analyze handwritten work and provide feedback.

    Supports vision models from Google, OpenAI, and Anthropic.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real API
    - delay: Add latency in milliseconds (0-30000)
    - error: Simulate error ("rate_limit", "timeout", "500")
    """
    log_id = request_logger.log_request(
        endpoint="/ai/feedback",
        mode=mode,
        prompt_preview=request_body.prompt[:100] if request_body.prompt else "",
        image_count=len(request_body.images),
        model=request_body.preferences.model,
    )

    try:
        if delay:
            await simulate_delay(delay)
        if error:
            simulate_error(error)

        # Mock mode
        if mode == "mock":
            mock_response = get_mock_response(
                endpoint="feedback",
                scenario=x_mock_scenario,
                prompt=request_body.prompt,
            )
            request_logger.log_response(log_id, success=True, mode="mock")
            return FeedbackResponse(
                feedback=mock_response,
                detected_content="Mock detected content",
                model=request_body.preferences.model,
                provider="mock",
                mode="mock",
            )

        # Production mode
        provider = provider_router.get_provider(request_body.preferences.model)

        prompt = build_feedback_prompt(
            custom_prompt=request_body.prompt,
            rag_context=request_body.rag_context,
            detail_level=request_body.preferences.detail_level,
            language=request_body.preferences.language,
        )

        config = GenerationConfig(
            temperature=0.7,
            max_tokens=4096,
        )

        # Convert images to provider format
        images = [{"data": img.data, "mime_type": img.mime_type} for img in request_body.images]

        result = await provider.generate_with_images(
            prompt=prompt,
            images=images,
            config=config,
        )

        request_logger.log_response(log_id, success=True, mode="prod")
        return FeedbackResponse(
            feedback=result.text,
            detected_content=None,
            model=result.model,
            provider=result.provider,
            mode="prod",
        )

    except HTTPException:
        raise
    except Exception as e:
        request_logger.log_response(log_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/quiz", response_model=QuizResponse)
async def ai_quiz(
    request_body: QuizRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
    delay: Optional[int] = Query(default=None, ge=0, le=30000),
    error: Optional[str] = Query(default=None, pattern="^(rate_limit|timeout|500)$"),
    x_mock_scenario: Optional[str] = Header(default=None, alias="X-Mock-Scenario"),
):
    """
    Generate quiz questions from course materials.

    Uses structured JSON output for reliable parsing.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real API
    - delay: Add latency in milliseconds (0-30000)
    - error: Simulate error ("rate_limit", "timeout", "500")
    """
    log_id = request_logger.log_request(
        endpoint="/ai/quiz",
        mode=mode,
        model=request_body.preferences.model,
        question_count=request_body.config.count,
    )

    try:
        if delay:
            await simulate_delay(delay)
        if error:
            simulate_error(error)

        # Mock mode
        if mode == "mock":
            mock_response = get_mock_response(
                endpoint="quiz",
                scenario=x_mock_scenario,
                prompt=request_body.rag_context.query,
            )
            # Parse mock response as JSON
            try:
                mock_data = json.loads(mock_response)
                questions = [
                    QuizQuestion(
                        id=q["id"],
                        type=q["type"],
                        question=q["question"],
                        options=[QuizOption(**opt) for opt in q.get("options", [])] if q.get("options") else None,
                        correct_answer=q["correct_answer"],
                        explanation=q["explanation"],
                        source_chunk_ids=q.get("source_chunk_ids", []),
                    )
                    for q in mock_data.get("questions", [])
                ]
            except json.JSONDecodeError:
                questions = []

            request_logger.log_response(log_id, success=True, mode="mock")
            return QuizResponse(
                questions=questions,
                model=request_body.preferences.model,
                provider="mock",
                mode="mock",
            )

        # Production mode
        provider = provider_router.get_provider(request_body.preferences.model)

        prompt = build_quiz_prompt(
            rag_context=request_body.rag_context,
            config=request_body.config,
            additional_instructions=request_body.additional_instructions,
        )

        config = GenerationConfig(
            temperature=0,
            max_tokens=8192,
            json_output=True,
            schema=QUIZ_RESPONSE_SCHEMA,
        )

        result = await provider.generate(prompt=prompt, config=config)

        # Parse the JSON response
        try:
            response_data = json.loads(result.text)
            questions = [
                QuizQuestion(
                    id=q["id"],
                    type=q["type"],
                    question=q["question"],
                    options=[QuizOption(**opt) for opt in q.get("options", [])] if q.get("options") else None,
                    correct_answer=q["correct_answer"],
                    explanation=q["explanation"],
                    source_chunk_ids=q.get("source_chunk_ids", []),
                )
                for q in response_data.get("questions", [])
            ]
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse quiz response: {e}")

        request_logger.log_response(log_id, success=True, mode="prod")
        return QuizResponse(
            questions=questions,
            model=result.model,
            provider=result.provider,
            mode="prod",
        )

    except HTTPException:
        raise
    except Exception as e:
        request_logger.log_response(log_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/chat", response_model=ChatResponse)
async def ai_chat(
    request_body: ChatRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
    delay: Optional[int] = Query(default=None, ge=0, le=30000),
    error: Optional[str] = Query(default=None, pattern="^(rate_limit|timeout|500)$"),
    x_mock_scenario: Optional[str] = Header(default=None, alias="X-Mock-Scenario"),
):
    """
    RAG-powered chat with course materials.

    Supports conversation history for multi-turn conversations.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real API
    - delay: Add latency in milliseconds (0-30000)
    - error: Simulate error ("rate_limit", "timeout", "500")
    """
    log_id = request_logger.log_request(
        endpoint="/ai/chat",
        mode=mode,
        prompt_preview=request_body.message[:100],
        model=request_body.preferences.model,
        history_length=len(request_body.conversation_history),
    )

    try:
        if delay:
            await simulate_delay(delay)
        if error:
            simulate_error(error)

        # Mock mode
        if mode == "mock":
            mock_response = get_mock_response(
                endpoint="chat",
                scenario=x_mock_scenario,
                prompt=request_body.message,
            )
            request_logger.log_response(log_id, success=True, mode="mock")

            # Build mock sources from context if available
            sources = []
            if request_body.rag_context:
                for chunk in request_body.rag_context.chunks[:3]:
                    sources.append(SourceReference(
                        source_id=chunk.source_id,
                        source_type=chunk.source_type,
                        heading=chunk.heading,
                        relevance=chunk.similarity,
                    ))

            return ChatResponse(
                message=mock_response,
                sources=sources,
                model=request_body.preferences.model,
                provider="mock",
                mode="mock",
            )

        # Production mode
        provider = provider_router.get_provider(request_body.preferences.model)

        prompt = build_chat_single_prompt(
            message=request_body.message,
            rag_context=request_body.rag_context,
            conversation_history=request_body.conversation_history,
            detail_level=request_body.preferences.detail_level,
            language=request_body.preferences.language,
        )

        config = GenerationConfig(
            temperature=0.7,
            max_tokens=4096,
        )

        result = await provider.generate(prompt=prompt, config=config)

        # Build sources from RAG context
        sources = []
        if request_body.rag_context:
            for chunk in request_body.rag_context.chunks[:5]:
                sources.append(SourceReference(
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    heading=chunk.heading,
                    relevance=chunk.similarity,
                ))

        request_logger.log_response(log_id, success=True, mode="prod")
        return ChatResponse(
            message=result.text,
            sources=sources,
            model=result.model,
            provider=result.provider,
            mode="prod",
        )

    except HTTPException:
        raise
    except Exception as e:
        request_logger.log_response(log_id, success=False, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
