"""Prompt templates for AI endpoints."""

from typing import Optional
from lib.models.common import RAGContext, DetailLevel
from lib.models.quiz import QuizConfig, QuestionType, Difficulty


def format_rag_context(context: Optional[RAGContext]) -> str:
    """Format RAG context for inclusion in prompts."""
    if not context or not context.chunks:
        return ""

    sections = []
    for chunk in context.chunks:
        source_label = f"{chunk.source_type.title()}"
        if chunk.heading:
            source_label += f" - {chunk.heading}"
        if chunk.page_number:
            source_label += f" (Page {chunk.page_number})"

        sections.append(f"---\n[{source_label}]\n{chunk.text}")

    context_text = "\n\n".join(sections)

    return f"""The following is relevant context from the student's course materials:

{context_text}
---

Use this context to provide accurate, relevant assistance. Reference specific sections when helpful."""


def build_feedback_prompt(
    custom_prompt: Optional[str],
    rag_context: Optional[RAGContext],
    detail_level: DetailLevel,
    language: str,
) -> str:
    """Build prompt for handwriting feedback endpoint."""

    detail_instructions = {
        DetailLevel.CONCISE: "Keep your feedback brief and focused on the most important points.",
        DetailLevel.BALANCED: "Provide clear, helpful feedback with appropriate detail.",
        DetailLevel.DETAILED: "Provide comprehensive feedback with detailed explanations and suggestions.",
    }

    language_instruction = ""
    if language != "en":
        language_instruction = f"\n\nRespond in {language}."

    context_section = format_rag_context(rag_context)
    if context_section:
        context_section = f"\n\n{context_section}"

    base_prompt = f"""You are an educational AI assistant helping a student with their handwritten work.

Analyze the handwritten content in the image(s) and provide constructive feedback.{context_section}

{detail_instructions[detail_level]}{language_instruction}

Guidelines:
- First, transcribe or describe what you see in the handwriting
- Identify the subject matter (math, science, writing, etc.)
- Provide feedback on correctness, clarity, and completeness
- Offer specific suggestions for improvement
- Be encouraging while pointing out areas for growth
- If the work contains errors, explain why they are incorrect and show the correct approach"""

    if custom_prompt:
        base_prompt += f"\n\nAdditional context from the student: {custom_prompt}"

    return base_prompt


def build_quiz_prompt(
    rag_context: RAGContext,
    config: QuizConfig,
    additional_instructions: Optional[str],
) -> str:
    """Build prompt for quiz generation endpoint."""

    context_section = format_rag_context(rag_context)

    question_type_descriptions = {
        QuestionType.MULTIPLE_CHOICE: "multiple choice questions with 4 options (A, B, C, D)",
        QuestionType.TRUE_FALSE: "true/false questions",
        QuestionType.SHORT_ANSWER: "short answer questions requiring a brief written response",
        QuestionType.FILL_BLANK: "fill-in-the-blank questions",
    }

    types_text = ", ".join(question_type_descriptions[t] for t in config.types)

    difficulty_instructions = {
        Difficulty.EASY: "Create straightforward questions that test basic understanding and recall.",
        Difficulty.MEDIUM: "Create questions that require application of concepts and some reasoning.",
        Difficulty.HARD: "Create challenging questions that require deep understanding, analysis, and synthesis.",
    }

    prompt = f"""You are an educational AI assistant creating a quiz based on course materials.

{context_section}

Generate exactly {config.count} quiz questions based on the context above.

Question Requirements:
- Question types to include: {types_text}
- Difficulty level: {config.difficulty.value}
- {difficulty_instructions[config.difficulty]}

For each question:
1. Create a clear, unambiguous question
2. For multiple choice: provide exactly 4 options labeled A, B, C, D
3. For true/false: the answer should be "True" or "False"
4. Provide the correct answer
5. Write an educational explanation of why the answer is correct
6. Reference which part of the source material the question is based on

Important:
- Questions should directly test understanding of the provided context
- Avoid trick questions or overly complex wording
- Ensure all correct answers are factually accurate based on the source material
- Distribute questions across different topics in the context when possible"""

    if additional_instructions:
        prompt += f"\n\nAdditional instructions: {additional_instructions}"

    prompt += """

Respond with a JSON object containing a "questions" array. Each question should have:
- id: unique identifier (q1, q2, etc.)
- type: question type
- question: the question text
- options: array of {label, text} for multiple choice (omit for other types)
- correct_answer: the correct answer
- explanation: why this is correct
- source_chunk_ids: array of source identifiers (can be empty)"""

    return prompt


def build_chat_prompt(
    message: str,
    rag_context: Optional[RAGContext],
    conversation_history: list,
    detail_level: DetailLevel,
    language: str,
) -> tuple[str, list[dict]]:
    """
    Build prompt for RAG chat endpoint.

    Returns:
        Tuple of (system_prompt, messages)
    """

    context_section = format_rag_context(rag_context)

    language_instruction = ""
    if language != "en":
        language_instruction = f" Respond in {language}."

    system_prompt = f"""You are a helpful educational AI assistant helping a student learn from their course materials.

{context_section if context_section else "No specific course context has been provided for this conversation."}

Guidelines:
- Answer questions accurately based on the provided context when available
- If the context doesn't contain relevant information, say so and provide general guidance
- Be encouraging and supportive
- Explain concepts clearly and provide examples when helpful
- When referencing specific parts of the context, mention the source{language_instruction}"""

    # Build messages array
    messages = []
    for msg in conversation_history:
        messages.append({
            "role": msg.role.value,
            "content": msg.content
        })

    # Add the current message
    messages.append({
        "role": "user",
        "content": message
    })

    return system_prompt, messages


def build_chat_single_prompt(
    message: str,
    rag_context: Optional[RAGContext],
    conversation_history: list,
    detail_level: DetailLevel,
    language: str,
) -> str:
    """
    Build a single prompt string for providers that don't support system prompts well.

    Combines system prompt and conversation into one string.
    """
    system_prompt, messages = build_chat_prompt(
        message, rag_context, conversation_history, detail_level, language
    )

    # Build the full prompt
    full_prompt = f"{system_prompt}\n\n"

    for msg in messages[:-1]:  # All but the last message (current)
        role = "Student" if msg["role"] == "user" else "Assistant"
        full_prompt += f"{role}: {msg['content']}\n\n"

    # Add current message
    full_prompt += f"Student: {message}\n\nAssistant:"

    return full_prompt
