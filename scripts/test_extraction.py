#!/usr/bin/env python3
"""
Test script for question extraction pipeline.
Saves extracted PDFs to documents/extractions/<datetime>/
"""

import base64
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.question_extractor import QuestionExtractor
from lib.latex_compiler import LaTeXCompiler


async def extract_and_save(pdf_path: str):
    """Extract questions from PDF and save to subdirectory named after the PDF."""

    # Create output directory named after the PDF (without extension)
    pdf_name = Path(pdf_path).stem
    output_dir = Path("documents/extractions") / pdf_name

    # Remove existing directory if it exists (to replace previous results)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Read PDF
    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    # Extract questions
    print("\n1. Extracting questions with Marker + Gemini...")
    extractor = QuestionExtractor()
    questions = await extractor.extract_questions(pdf_b64)
    print(f"   Found {len(questions)} questions")

    # Compile and save each question
    print("\n2. Compiling to PDF with Tectonic...")
    compiler = LaTeXCompiler()

    for q in questions:
        try:
            compiled = compiler.compile_question(
                order_index=q.order_index,
                question_number=q.question_number,
                latex_content=q.latex_content,
                has_images=q.has_images,
                has_tables=q.has_tables,
                image_data=q.image_data if q.image_data else None
            )

            # Save PDF
            safe_name = q.question_number.replace(" ", "_").replace("/", "-")
            filename = f"{q.order_index + 1:02d}_{safe_name}.pdf"
            pdf_bytes = base64.b64decode(compiled.pdf_base64)

            output_path = output_dir / filename
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)

            print(f"   ✓ {filename} ({len(pdf_bytes)} bytes)")

        except Exception as e:
            print(f"   ✗ {q.question_number}: {e}")

    print(f"\n✓ Done! Files saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the test PDF
        pdf_path = "documents/Problem_Set_1.pdf"
    else:
        pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"Processing: {pdf_path}")
    asyncio.run(extract_and_save(pdf_path))
