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
    # AI endpoint responses
    "feedback": {
        "default": "I can see your handwritten work. Your approach shows good understanding of the concept. Here are some suggestions:\n\n1. Your notation is clear and well-organized\n2. The calculation in step 2 is correct\n3. Consider double-checking your final answer by substituting back into the original equation\n\nOverall, great effort! Keep practicing to build confidence.",
        "math": "Looking at your mathematical work:\n\n**Detected content:** You've written the quadratic formula and are solving for x.\n\n**Feedback:**\n- Your setup is correct: $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$\n- Watch the sign in the discriminant calculation\n- Your final answers look good!\n\nExcellent work on showing all your steps.",
        "error": "I notice an error in your work. In step 3, when you multiplied both sides by 2, you forgot to distribute to the second term. Let me help:\n\nOriginal: $\\frac{x+3}{2} = 5$\nCorrect: $x + 3 = 10$\n\nFrom here, you would get $x = 7$, not $x = 4$.",
    },
    "quiz": {
        "default": json.dumps({
            "questions": [
                {
                    "id": "q1",
                    "type": "multiple_choice",
                    "question": "What is the derivative of x^2?",
                    "options": [
                        {"label": "A", "text": "x"},
                        {"label": "B", "text": "2x"},
                        {"label": "C", "text": "2"},
                        {"label": "D", "text": "x^2"}
                    ],
                    "correct_answer": "B",
                    "explanation": "Using the power rule, d/dx(x^n) = nx^(n-1). For x^2, this gives 2x^1 = 2x.",
                    "source_chunk_ids": ["chunk_1"]
                },
                {
                    "id": "q2",
                    "type": "true_false",
                    "question": "The integral of a constant is always zero.",
                    "correct_answer": "False",
                    "explanation": "The integral of a constant c is cx + C, where C is the constant of integration. It's only zero when c = 0.",
                    "source_chunk_ids": ["chunk_2"]
                },
                {
                    "id": "q3",
                    "type": "multiple_choice",
                    "question": "Which of the following is the chain rule formula?",
                    "options": [
                        {"label": "A", "text": "d/dx[f(g(x))] = f'(x)g'(x)"},
                        {"label": "B", "text": "d/dx[f(g(x))] = f'(g(x)) * g'(x)"},
                        {"label": "C", "text": "d/dx[f(g(x))] = f(g'(x))"},
                        {"label": "D", "text": "d/dx[f(g(x))] = f'(g(x)) + g'(x)"}
                    ],
                    "correct_answer": "B",
                    "explanation": "The chain rule states that the derivative of a composite function f(g(x)) is the derivative of the outer function evaluated at the inner function, multiplied by the derivative of the inner function.",
                    "source_chunk_ids": ["chunk_1", "chunk_3"]
                }
            ]
        }),
    },
    "chat": {
        "default": "Based on the course materials, I can help you understand this concept. The key idea is that derivatives measure the rate of change of a function. When you see a problem asking for the derivative, you're essentially finding how fast the function's output changes as the input changes.\n\nIs there a specific part of this concept you'd like me to explain further?",
        "explain": "Let me explain this step by step:\n\n1. **First**, identify the type of function you're working with\n2. **Then**, choose the appropriate differentiation rule\n3. **Finally**, apply the rule and simplify\n\nFrom your course notes on Chapter 3, the power rule is: $\\frac{d}{dx}x^n = nx^{n-1}$\n\nWould you like me to walk through a specific example?",
        "hint": "I see you're working on integration by parts. Here's a hint without giving away the full answer:\n\nRemember the formula: $\\int u \\, dv = uv - \\int v \\, du$\n\nFor this problem, try letting $u = x$ and $dv = e^x dx$. What does that give you for $du$ and $v$?",
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
