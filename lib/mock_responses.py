"""
Mock Responses - Predefined responses for testing without hitting the real API.

Provides realistic mock responses for different scenarios.
"""

from typing import Optional
import json

# Mock responses for different scenarios
MOCK_RESPONSES = {
    # Default responses by endpoint
    "generate": {
        "default": "This is a mock response from the Reef server. The prompt was received and processed successfully.",
        "latex": r"The equation can be written as: $\frac{d}{dx}(x^2) = 2x$",
        "json": json.dumps({
            "result": "success",
            "data": {"message": "Mock JSON response"}
        }),
        "feedback": "Great work on this problem! Your approach is correct. Consider reviewing the distributive property for the next step.",
        "transcription": r"$\int_0^1 x^2 dx = \frac{1}{3}$",
        "error_format": "I couldn't process that request. Please try again with a clearer prompt.",
    },
    "vision": {
        "default": "This is a mock vision response. I can see the image you've provided.",
        "handwriting": r"The handwritten text appears to be: $y = mx + b$ where m is the slope and b is the y-intercept.",
        "diagram": "I can see a diagram showing a right triangle with sides labeled a, b, and c. This appears to be related to the Pythagorean theorem.",
        "equation": r"The written equation is: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$",
        "problem": json.dumps({
            "problem_type": "calculus",
            "expression": r"\frac{d}{dx}(3x^2 + 2x)",
            "steps": [
                "Apply power rule to 3x^2: 6x",
                "Apply power rule to 2x: 2",
                "Combine: 6x + 2"
            ],
            "answer": "6x + 2"
        }),
    },
}

# Scenario-specific responses (used with X-Mock-Scenario header)
SCENARIO_RESPONSES = {
    "latex_simple": r"$x + y = z$",
    "latex_complex": r"$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot d\mathbf{S}$",
    "empty": "",
    "long": "This is a very long response. " * 100,
    "json_array": json.dumps([
        {"id": 1, "text": "First item"},
        {"id": 2, "text": "Second item"},
        {"id": 3, "text": "Third item"},
    ]),
    "feedback_positive": "Excellent work! Your solution is correct and well-organized.",
    "feedback_hint": "You're on the right track, but check your calculation in step 3.",
    "feedback_wrong": "That's not quite right. Let's review the concept together.",
}


def get_mock_response(
    endpoint: str,
    scenario: Optional[str] = None,
    prompt: Optional[str] = None,
) -> str:
    """
    Get a mock response for testing.

    Args:
        endpoint: The endpoint being called ("generate" or "vision")
        scenario: Optional specific scenario from X-Mock-Scenario header
        prompt: The original prompt (used for smart response selection)

    Returns:
        Mock response text
    """
    # If a specific scenario is requested, use it
    if scenario and scenario in SCENARIO_RESPONSES:
        return SCENARIO_RESPONSES[scenario]

    # Get endpoint-specific responses
    endpoint_responses = MOCK_RESPONSES.get(endpoint, MOCK_RESPONSES["generate"])

    # Try to select appropriate response based on prompt content
    if prompt:
        prompt_lower = prompt.lower()

        if "latex" in prompt_lower or "equation" in prompt_lower:
            return endpoint_responses.get("latex", endpoint_responses.get("equation", endpoint_responses["default"]))

        if "json" in prompt_lower or "structured" in prompt_lower:
            return endpoint_responses.get("json", endpoint_responses.get("problem", endpoint_responses["default"]))

        if "feedback" in prompt_lower or "review" in prompt_lower:
            return endpoint_responses.get("feedback", endpoint_responses["default"])

        if "transcri" in prompt_lower or "handwrit" in prompt_lower:
            return endpoint_responses.get("transcription", endpoint_responses.get("handwriting", endpoint_responses["default"]))

        if "diagram" in prompt_lower or "image" in prompt_lower:
            return endpoint_responses.get("diagram", endpoint_responses["default"])

    # Default response
    return endpoint_responses["default"]


def list_scenarios() -> list[str]:
    """List all available mock scenarios."""
    return list(SCENARIO_RESPONSES.keys())
