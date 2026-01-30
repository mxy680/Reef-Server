"""
Request Logger - Logs API requests for debugging and monitoring.

Stores recent requests in memory for development inspection.
"""

from datetime import datetime, timezone
from typing import Optional
from collections import deque
import uuid


class RequestLogger:
    """In-memory request logger for development debugging."""

    def __init__(self, max_logs: int = 1000):
        """
        Initialize the logger.

        Args:
            max_logs: Maximum number of logs to retain in memory
        """
        self._logs: deque = deque(maxlen=max_logs)

    def log_request(
        self,
        endpoint: str,
        mode: str,
        prompt_preview: str = "",
        image_count: int = 0,
    ) -> str:
        """
        Log an incoming request.

        Args:
            endpoint: The API endpoint called
            mode: "mock" or "prod"
            prompt_preview: First 100 chars of prompt
            image_count: Number of images in request (for vision)

        Returns:
            Log ID for correlating with response
        """
        log_id = str(uuid.uuid4())[:8]

        log_entry = {
            "id": log_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "mode": mode,
            "prompt_preview": prompt_preview,
            "image_count": image_count,
            "status": "pending",
            "response_time_ms": None,
            "error": None,
        }

        self._logs.append(log_entry)
        return log_id

    def log_response(
        self,
        log_id: str,
        success: bool,
        mode: str = "",
        error: Optional[str] = None,
    ) -> None:
        """
        Update a log entry with response info.

        Args:
            log_id: The log ID from log_request
            success: Whether the request succeeded
            mode: The mode used ("mock" or "prod")
            error: Error message if failed
        """
        for log in reversed(self._logs):
            if log["id"] == log_id:
                request_time = datetime.fromisoformat(log["timestamp"])
                response_time = datetime.now(timezone.utc)
                log["response_time_ms"] = int(
                    (response_time - request_time).total_seconds() * 1000
                )
                log["status"] = "success" if success else "error"
                if mode:
                    log["mode"] = mode
                if error:
                    log["error"] = error
                break

    def get_logs(self, limit: int = 100) -> list[dict]:
        """
        Get recent logs.

        Args:
            limit: Maximum number of logs to return

        Returns:
            List of log entries, most recent first
        """
        logs = list(self._logs)
        logs.reverse()
        return logs[:limit]

    def clear_logs(self) -> None:
        """Clear all logs."""
        self._logs.clear()
