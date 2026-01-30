"""
Simulator - Simulate network conditions and errors for testing.

Provides latency injection and error simulation.
"""

import asyncio
from fastapi import HTTPException


async def simulate_delay(delay_ms: int) -> None:
    """
    Simulate network latency.

    Args:
        delay_ms: Delay in milliseconds (0-30000)
    """
    if delay_ms > 0:
        await asyncio.sleep(delay_ms / 1000)


def simulate_error(error_type: str) -> None:
    """
    Simulate an API error.

    Args:
        error_type: Type of error to simulate:
            - "rate_limit": HTTP 429 Too Many Requests
            - "timeout": HTTP 504 Gateway Timeout
            - "500": HTTP 500 Internal Server Error

    Raises:
        HTTPException: With appropriate status code and message
    """
    if error_type == "rate_limit":
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Please retry after 60 seconds.",
                "retry_after": 60,
            }
        )

    elif error_type == "timeout":
        raise HTTPException(
            status_code=504,
            detail={
                "error": "gateway_timeout",
                "message": "The upstream server did not respond in time.",
            }
        )

    elif error_type == "500":
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
            }
        )


# Error scenarios for more complex testing
ERROR_SCENARIOS = {
    "rate_limit": {
        "status_code": 429,
        "error": "rate_limit_exceeded",
        "message": "Rate limit exceeded. Please retry after 60 seconds.",
    },
    "timeout": {
        "status_code": 504,
        "error": "gateway_timeout",
        "message": "The upstream server did not respond in time.",
    },
    "500": {
        "status_code": 500,
        "error": "internal_server_error",
        "message": "An unexpected error occurred. Please try again later.",
    },
    "auth_failed": {
        "status_code": 401,
        "error": "authentication_failed",
        "message": "Invalid API key provided.",
    },
    "quota_exceeded": {
        "status_code": 403,
        "error": "quota_exceeded",
        "message": "API quota exceeded for this billing period.",
    },
    "model_unavailable": {
        "status_code": 503,
        "error": "model_unavailable",
        "message": "The requested model is temporarily unavailable.",
    },
}


def get_error_scenarios() -> list[str]:
    """List all available error scenarios."""
    return list(ERROR_SCENARIOS.keys())
