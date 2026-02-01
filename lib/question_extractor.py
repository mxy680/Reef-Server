"""
Question extraction service using Marker for OCR and Gemini for segmentation.

Pipeline:
1. Marker extracts markdown + images from PDF
2. Gemini segments questions and formats as LaTeX
3. Each question is compiled to PDF via tectonic
"""

import os
import json
import tempfile
import base64
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import google.generativeai as genai


@dataclass
class ExtractedQuestion:
    """A single extracted question ready for compilation."""
    order_index: int
    question_number: str
    latex_content: str
    has_images: bool
    has_tables: bool
    image_paths: list[str]


QUESTION_SEGMENTATION_PROMPT = """You are an expert at analyzing educational documents (exams, study guides, problem sets).

Given the following markdown content extracted from a PDF, identify and segment individual questions.

For each question:
1. Identify the question number (e.g., "1", "2a", "Problem 3")
2. Keep sub-questions (a, b, c, i, ii, etc.) together with their parent question
3. Convert the content to clean LaTeX format
4. Include any images using \\includegraphics{{image_name}}
5. Format tables using the booktabs package (\\toprule, \\midrule, \\bottomrule)
6. Format math expressions using proper LaTeX math mode ($ for inline, $$ or \\[ \\] for display)

Return a JSON array with objects containing:
- "question_number": string (the question identifier)
- "latex_content": string (the LaTeX content for this question)
- "has_images": boolean (true if images are referenced)
- "has_tables": boolean (true if tables are present)
- "image_refs": array of strings (image filenames referenced)

IMPORTANT:
- Return ONLY valid JSON, no markdown code blocks
- Each question should be self-contained
- Preserve all mathematical notation accurately
- Keep multi-part questions together

Markdown content:
{markdown_content}

Image references available: {image_refs}
"""


class QuestionExtractor:
    """Extracts questions from PDFs using Marker and Gemini."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        """Initialize the extractor with Gemini API key."""
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

    def extract_from_pdf(self, pdf_path: str, output_dir: str) -> tuple[str, list[str]]:
        """
        Extract markdown and images from PDF using Marker.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images

        Returns:
            Tuple of (markdown_content, list of image paths)
        """
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError:
            raise ImportError(
                "marker-pdf is required. Install with: pip install marker-pdf"
            )

        # Initialize Marker converter
        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)

        # Convert PDF to markdown
        rendered = converter(pdf_path)
        markdown_text, _, images = text_from_rendered(rendered)

        # Save images to output directory
        image_paths = []
        images_dir = Path(output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for img_name, img_data in images.items():
            img_path = images_dir / img_name
            with open(img_path, "wb") as f:
                f.write(img_data)
            image_paths.append(str(img_path))

        return markdown_text, image_paths

    def segment_questions(
        self,
        markdown_content: str,
        image_refs: list[str]
    ) -> list[ExtractedQuestion]:
        """
        Use Gemini to segment questions and format as LaTeX.

        Args:
            markdown_content: Markdown text from Marker
            image_refs: List of image filenames

        Returns:
            List of ExtractedQuestion objects
        """
        # Prepare the prompt
        image_ref_str = ", ".join(image_refs) if image_refs else "none"
        prompt = QUESTION_SEGMENTATION_PROMPT.format(
            markdown_content=markdown_content,
            image_refs=image_ref_str
        )

        # Call Gemini
        response = self.model.generate_content(prompt)
        response_text = response.text.strip()

        # Clean up response if wrapped in code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines if they're code block markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)

        # Parse JSON response
        try:
            questions_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini response as JSON: {e}")

        # Convert to ExtractedQuestion objects
        questions = []
        for idx, q in enumerate(questions_data):
            questions.append(ExtractedQuestion(
                order_index=idx,
                question_number=q.get("question_number", str(idx + 1)),
                latex_content=q.get("latex_content", ""),
                has_images=q.get("has_images", False),
                has_tables=q.get("has_tables", False),
                image_paths=q.get("image_refs", [])
            ))

        return questions

    async def extract_questions(self, pdf_base64: str) -> list[ExtractedQuestion]:
        """
        Full pipeline: PDF -> Marker -> Gemini -> ExtractedQuestions.

        Args:
            pdf_base64: Base64-encoded PDF content

        Returns:
            List of ExtractedQuestion objects ready for LaTeX compilation
        """
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Decode and save PDF
            pdf_bytes = base64.b64decode(pdf_base64)
            pdf_path = Path(temp_dir) / "input.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)

            # Extract with Marker
            markdown_content, image_paths = self.extract_from_pdf(
                str(pdf_path),
                temp_dir
            )

            # Get just the image filenames for Gemini
            image_refs = [Path(p).name for p in image_paths]

            # Segment with Gemini
            questions = self.segment_questions(markdown_content, image_refs)

            # Update image paths to full paths
            for q in questions:
                q.image_paths = [
                    str(Path(temp_dir) / "images" / name)
                    for name in q.image_paths
                    if (Path(temp_dir) / "images" / name).exists()
                ]

            return questions
