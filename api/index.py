"""
Reef Server - FastAPI application for PDF reconstruction and text embeddings.

Provides:
- PDF reconstruction pipeline (Surya layout → Gemini grouping → OpenAI extraction → LaTeX)
- Text embeddings using MiniLM-L6-v2
- PDF layout annotation using Surya
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import json
import os
import io
import asyncio
import base64
from pathlib import Path
from datetime import datetime
import time

# Import lib modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from lib.mock_responses import get_mock_embedding
import re
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np
from lib.models import EmbedRequest, EmbedResponse, ProblemGroup, GroupProblemsResponse, Question, QuestionBatch, QuizGenerationRequest, QuizQuestionResponse
from lib.embedding_client import get_embedding_service
from lib.question_to_latex import question_to_latex, quiz_question_to_latex, _sanitize_text
from lib.database import init_db, close_db
from api.users import router as users_router
from api.tts import router as tts_router
from api.strokes import router as strokes_router

from lib.surya_client import detect_layout
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database at startup."""
    print("[Startup] Initializing...")

    # Initialize database (no-op if DATABASE_URL not set)
    await init_db()

    print("[Startup] Ready! (Surya + TTS + Embeddings on Modal)")

    yield

    print("[Shutdown] Cleaning up...")
    await close_db()


app = FastAPI(
    title="Reef Server",
    description="PDF reconstruction and embedding service for Reef iOS app",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users_router)
app.include_router(tts_router)
app.include_router(strokes_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "reef-server",
        "version": "1.0.0"
    }


@app.post("/ai/embed", response_model=EmbedResponse)
async def ai_embed(
    request_body: EmbedRequest,
    request: Request,
    mode: str = Query(default="prod", pattern="^(mock|prod)$"),
):
    """
    Generate text embeddings using MiniLM-L6-v2.

    Accepts a single text string or a list of texts. Returns 384-dimensional
    normalized vectors suitable for semantic search.

    Query Parameters:
    - mode: "mock" for testing, "prod" for real embeddings
    """
    # Normalize input to list
    texts = request_body.texts if isinstance(request_body.texts, list) else [request_body.texts]
    text_count = len(texts)

    try:
        # Validate input
        if text_count == 0:
            raise HTTPException(status_code=422, detail="texts cannot be empty")
        if text_count > 100:
            raise HTTPException(status_code=422, detail="Maximum 100 texts per request")

        # Mock mode
        if mode == "mock":
            embeddings = get_mock_embedding(count=text_count, dimensions=384)
            return EmbedResponse(
                embeddings=embeddings,
                model="all-MiniLM-L6-v2",
                dimensions=384,
                count=text_count,
                mode="mock",
            )

        # Production mode
        embedding_service = get_embedding_service()
        embeddings = embedding_service.embed(texts, normalize=request_body.normalize)

        return EmbedResponse(
            embeddings=embeddings,
            model=embedding_service.model_name,
            dimensions=embedding_service.dimensions,
            count=text_count,
            mode="prod",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@dataclass
class SyntheticBox:
    """Mimics Surya's LayoutBox interface for injected detections."""
    bbox: list[float]
    label: str


def detect_uncovered_figures(page_img: Image.Image, layout_bboxes) -> list[list[float]]:
    """Detect visual content not covered by any Surya bounding box.

    Converts the page to grayscale, masks out known bboxes and margins,
    then finds clusters of remaining non-white pixels using a block grid + BFS.
    Returns bounding boxes for clusters large enough to be figures.
    """
    BLOCK = 30
    PADDING = 15
    DENSITY_THRESH = 0.02
    MIN_W, MIN_H = 80, 80

    gray = np.array(page_img.convert("L"))
    h, w = gray.shape

    # Mask out all Surya-detected regions (+ padding)
    for box in layout_bboxes:
        bx1 = max(0, int(box.bbox[0]) - PADDING)
        by1 = max(0, int(box.bbox[1]) - PADDING)
        bx2 = min(w, int(box.bbox[2]) + PADDING)
        by2 = min(h, int(box.bbox[3]) + PADDING)
        gray[by1:by2, bx1:bx2] = 255

    # Blank page margins (top/bottom 4%, left/right 3%)
    margin_tb = int(h * 0.04)
    margin_lr = int(w * 0.03)
    gray[:margin_tb, :] = 255
    gray[h - margin_tb:, :] = 255
    gray[:, :margin_lr] = 255
    gray[:, w - margin_lr:] = 255

    # Compute block density grid
    rows = h // BLOCK
    cols = w // BLOCK
    active = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            patch = gray[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK]
            density = np.mean(patch < 230)
            if density > DENSITY_THRESH:
                active[r, c] = True

    # BFS connected components on active blocks
    visited = np.zeros_like(active)
    results: list[list[float]] = []
    for r in range(rows):
        for c in range(cols):
            if active[r, c] and not visited[r, c]:
                queue = deque([(r, c)])
                visited[r, c] = True
                min_r, max_r, min_c, max_c = r, r, c, c
                while queue:
                    cr, cc = queue.popleft()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and active[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                # Convert block coords back to pixel coords
                x1 = min_c * BLOCK
                y1 = min_r * BLOCK
                x2 = (max_c + 1) * BLOCK
                y2 = (max_r + 1) * BLOCK
                if (x2 - x1) >= MIN_W and (y2 - y1) >= MIN_H:
                    results.append([float(x1), float(y1), float(x2), float(y2)])

    return results


def annotate_page(img: Image.Image, layout_result, scale: int = 2, start_index: int = 1) -> tuple[Image.Image, int]:
    """Annotate a single page image with layout detection results.

    Returns the annotated image and the next index to use.
    """
    # Scale up for better output quality
    img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

    # Convert to RGBA for transparency support
    img = img.convert("RGBA")

    # Create overlay for semi-transparent fills
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Main draw for outlines and text
    draw = ImageDraw.Draw(img)

    # Use Pillow's built-in font at large size (works on all platforms)
    font = ImageFont.load_default(size=48)

    # All annotations in red
    rgb = (220, 53, 69)  # Red color

    current_index = start_index
    for block in layout_result.bboxes:
        # Scale bbox coordinates to match scaled image
        bbox = block.bbox
        x1 = int(bbox[0] * scale)
        y1 = int(bbox[1] * scale)
        x2 = int(bbox[2] * scale)
        y2 = int(bbox[3] * scale)

        # Draw semi-transparent fill on overlay
        fill_color = (*rgb, 40)  # 40/255 alpha
        overlay_draw.rectangle([(x1, y1), (x2, y2)], fill=fill_color)

        # Draw thick outline
        for i in range(3):
            draw.rectangle(
                [(x1 - i, y1 - i), (x2 + i, y2 + i)],
                outline=rgb,
                width=2
            )

        # Draw index label
        label = str(current_index)
        label_y = max(y1 - 60, 5)  # Don't go above image
        text_bbox = draw.textbbox((x1, label_y), label, font=font)
        padding = 10
        label_rect = (text_bbox[0] - padding, text_bbox[1] - padding,
                     text_bbox[2] + padding, text_bbox[3] + padding)
        draw.rectangle(label_rect, fill=rgb)
        draw.text((x1, label_y), label, fill="white", font=font)

        current_index += 1

    # Composite the overlay onto the image
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB
    return img.convert("RGB"), current_index


@app.post("/ai/annotate")
async def ai_annotate(
    pdf: UploadFile = File(..., description="PDF file to annotate"),
):
    """
    Annotate all pages of a PDF with layout detection using Surya.

    Draws colored bounding boxes around detected layout elements:
    - Text blocks (blue)
    - Titles (red)
    - Tables (orange)
    - Figures (pink)
    - And more...

    Returns the annotated PDF with all pages.
    """
    try:
        # Read PDF content
        pdf_bytes = await pdf.read()

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)

        # Convert all pages to images at 96 DPI (what Surya expects)
        images = []
        mat = fitz.Matrix(96/72, 96/72)
        for page_num in range(num_pages):
            pdf_page = doc[page_num]
            pix = pdf_page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()

        # Run Surya layout detection via Modal GPU
        layout_results = await asyncio.to_thread(detect_layout, images)

        # Annotate each page with continuous indexing
        annotated_pages = []
        current_index = 1
        for img, layout_result in zip(images, layout_results):
            annotated, current_index = annotate_page(img, layout_result, scale=2, start_index=current_index)
            annotated_pages.append(annotated)

        # Save to /data/annotations directory as PDF
        annotations_dir = Path(__file__).parent.parent / "data" / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename from original PDF name
        base_name = Path(pdf.filename).stem if pdf.filename else "document"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_annotated_{timestamp}.pdf"
        output_path = annotations_dir / output_filename

        # Save all pages as PDF
        if annotated_pages:
            annotated_pages[0].save(
                output_path,
                "PDF",
                save_all=True,
                append_images=annotated_pages[1:] if len(annotated_pages) > 1 else [],
                resolution=150,
            )

        # Return the PDF
        output = io.BytesIO()
        if annotated_pages:
            annotated_pages[0].save(
                output,
                "PDF",
                save_all=True,
                append_images=annotated_pages[1:] if len(annotated_pages) > 1 else [],
                resolution=150,
            )
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={output_filename}",
                "X-Saved-Path": str(output_path),
                "X-Page-Count": str(num_pages),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


GROUP_PROBLEMS_PROMPT = """\
You are analyzing scanned pages of a homework or assignment document.

Each page has been annotated with numbered red bounding boxes (indices 1 through {total_annotations}).
Each bounding box surrounds a detected layout element (text block, title, figure, table, formula, etc.).

Your task: identify EVERY numbered problem in the document and map each one to the annotation indices that contain it.

Rules:
- CRITICAL: You must find ALL numbered problems visible in the document. Read every page carefully and count every problem number you see. Do NOT skip any.
- Multiple problems often share the same annotation index — this is normal when a single bounding box contains several questions. Create a separate problem group for each problem number, even if they all point to the same annotation index.
- Use the visible problem numbers/identifiers in the document for problem_number.
- Only include annotations that belong to a specific numbered problem (question text, sub-parts, figures, formulas, tables, etc.).
- Pay special attention to figures and pictures — always assign them to the problem they illustrate. Figures usually appear directly above or below the problem text they belong to.
- Skip annotations that are general context: page headers, page footers, document titles, course info, general directions/instructions, answer keys/solutions sections, or any content not tied to a specific problem.
- Not every annotation index needs to appear — omit ones that aren't part of a problem.
- Use a short descriptive label for each group (e.g. "Problem 1", "Problem 2a-2c").
- Order the problems by their appearance in the document.

Return a JSON object matching the provided schema.
"""


@app.post("/ai/group-problems")
async def ai_group_problems(
    pdf: UploadFile = File(..., description="PDF file to annotate and group"),
):
    """
    Annotate a PDF with numbered bounding boxes, then use Gemini to group
    annotations into logical problem groups.

    Returns JSON with problem groups and their annotation indices.
    """
    try:
        # Read PDF content
        pdf_bytes = await pdf.read()

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)

        # Convert all pages to images at 96 DPI (what Surya expects)
        images = []
        mat = fitz.Matrix(96/72, 96/72)
        for page_num in range(num_pages):
            pdf_page = doc[page_num]
            pix = pdf_page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        doc.close()

        # Run Surya layout detection via Modal GPU
        layout_results = await asyncio.to_thread(detect_layout, images)

        # Annotate each page with continuous indexing
        annotated_pages = []
        current_index = 1
        for img, layout_result in zip(images, layout_results):
            annotated, current_index = annotate_page(img, layout_result, scale=2, start_index=current_index)
            annotated_pages.append(annotated)

        total_annotations = current_index - 1

        # Convert annotated pages to JPEG bytes for Gemini
        page_images: list[bytes] = []
        for page in annotated_pages:
            buf = io.BytesIO()
            page.save(buf, format="JPEG", quality=85)
            page_images.append(buf.getvalue())

        # Call Gemini (via OpenRouter) with annotated images and prompt
        from lib.llm_client import LLMClient
        prompt = GROUP_PROBLEMS_PROMPT.format(total_annotations=total_annotations)
        llm_client = LLMClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
        )
        raw_response = llm_client.generate(
            prompt=prompt,
            images=page_images,
            response_schema=GroupProblemsResponse.model_json_schema(),
        )

        # Parse and validate response
        result = GroupProblemsResponse.model_validate_json(raw_response)

        # Override computed fields with known values
        result.total_annotations = total_annotations
        result.total_problems = len(result.problems)
        result.page_count = num_pages

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


EXTRACT_QUESTION_PROMPT = """\
You are extracting structured question data from scanned homework/exam images.
Extract the content exactly as shown — do NOT solve problems, fill in blanks, or interpret the content.

The images have red numbered annotation boxes overlaid — ignore them.

## Structure
- number: The problem number as shown in the document.
- text: The question stem / preamble. For simple questions with no parts, all content goes here.
- figures: List of figure filenames that belong to the stem (from the list below, if any).
- parts: Labeled sub-questions (a, b, c). Parts can nest recursively (a → i, ii, iii).
  - If a question has unlabeled bullet points or numbered sub-items, use sequential letters (a, b, c...) as labels.
  - IMPORTANT: If a part contains multiple questions that each need a separate answer (e.g. Q1, Q2, Q3... or bullet points asking different things), extract each as a nested sub-part — do NOT combine them into a single \\begin{itemize} list. Each question that needs its own answer space must be its own part.

## CRITICAL: All text fields must be valid LaTeX body content

Every `text` field will be compiled by a LaTeX engine. You are responsible for producing text that compiles without errors.

Rules:
- Escape LaTeX special characters in prose: write \\& not &, \\% not %, \\# not #, \\$ not $ (when not math).
- Inline math: $...$ delimiters. Display math: \\[...\\].
- All LaTeX commands (\\Delta, \\sigma, \\rightarrow, \\text{}, \\frac{}{}, etc.) MUST be inside math delimiters.
- Degree symbols: $^\\circ$ (e.g. $100^\\circ$C). Never use raw ° or \\degree.
- Subscripts/superscripts: always in math mode ($H_2O$, $x^2$, $q_{\\text{rxn}}$).
- Bold text: use \\textbf{...}, NOT markdown **bold**.
- Itemized lists: use \\begin{itemize} \\item ... \\end{itemize}.
- Tables: use \\begin{tabular}{...} with proper & column separators and \\\\ row endings.
- NO Unicode symbols — use LaTeX equivalents (\\rightarrow not →, \\neq not ≠, \\leq not ≤, etc.).
- NO markdown syntax whatsoever.
- Combine all text for a section into a single string — do NOT split into separate blocks.

## Figures
- Figures are a list of filenames, NOT inline content.
- Place figure filenames at the level where they appear (question-level or part-level).

## Tables that define sub-questions
When a problem contains a table whose rows correspond to the labeled sub-parts (e.g. a table with rows a, b, c showing function pairs or data), preserve the table as a \\begin{tabular} in the stem text. The parts should then have EMPTY text (just the label and answer space) since the table already presents the content. Do NOT flatten table rows into separate part text fields — this loses the tabular formatting.

## Problem-specific figures (ALWAYS include)
If a figure or diagram is visually adjacent to a problem (directly above, below, or beside the question text), it almost certainly belongs to that problem. ALWAYS include these in the `figures` list — even if the question text never says "see figure" or "refer to the diagram." Use common sense: if a question asks you to identify, analyze, draw, or calculate something and there is a nearby figure (structural formula, graph, apparatus diagram, reaction scheme, molecule drawing, etc.) that would be needed to answer it, include it. The student cannot answer the question without seeing that figure.

## Referenced data from other pages (tables, models, formulas)
The images may include multiple pages. Some pages contain reference material (data tables, models, formulas, definitions) that problems refer to by name (e.g. "Table 1", "the model", "the equation above").

For reference data on OTHER pages (not adjacent to the problem):
- Only include it if the problem explicitly mentions it by name.
- If a problem does NOT mention "Table 1" or "the model" or similar, do NOT include that data.

When a problem DOES explicitly reference external data:
- **Tables:** Include the table image file (e.g. table_5.jpg) in the question's `figures` list. Do NOT reproduce tables as \\begin{tabular} — the image preserves visual content (structural formulas, drawings) that cannot be represented in LaTeX.
- **Models, equations, formulas:** Reproduce these in the question's `text` field as LaTeX.
- This applies to EVERY question that references the data. If problems 1 through 7 all say "Table 1", ALL seven must include the table image.

## Answer space
Estimate answer_space_cm at the most specific level (deepest part > parent part > question):
- 1.0: multiple choice / true-false / short factual
- 2.0: one-line calculation or brief explanation
- 3.0: standard calculation or paragraph
- 4.0: multi-step derivation or proof
- 6.0: long proof, graph to sketch, or multi-part calculation
"""


LATEX_FIX_PROMPT = """You are a LaTeX expert. The following LaTeX body content failed to compile. Fix it and return ONLY the corrected LaTeX body content — no preamble, no \\documentclass, no \\begin{{document}}.

## Failed LaTeX
```
{latex_body}
```

## Compilation Error
```
{error_message}
```

## Rules
- Return ONLY the fixed LaTeX body content, nothing else
- Do NOT wrap in code fences or markdown
- Do NOT add \\documentclass, \\usepackage, \\begin{{document}}, or \\end{{document}}
- Fix the specific error shown above
- Preserve all content — do not remove or simplify questions
- Keep all math in $...$ or \\[...\\] delimiters
- Available packages: amsmath, amssymb, amsfonts, graphicx, booktabs, array, xcolor, needspace, algorithm, algorithmic, listings, caption, changepage
"""


@app.post("/ai/reconstruct")
async def ai_reconstruct(
    pdf: UploadFile = File(..., description="PDF file to reconstruct"),
    debug: bool = Query(default=False, description="Save intermediate files to data/"),
    split: bool = Query(default=False, description="Return individual problem PDFs as JSON instead of merged PDF"),
):
    """
    Reconstruct a homework PDF into cleanly typeset LaTeX.

    Pipeline:
    1. Annotate PDF pages with Surya layout detection
    2. Group annotations into problems via Gemini
    3. Crop each problem's regions from original pages
    4. Send cropped regions to Gemini in parallel for LaTeX reconstruction
    5. Compile each problem's LaTeX to PDF via tectonic
    6. Merge all per-problem PDFs into one final PDF
    """
    try:
        # Read PDF content
        pdf_bytes = await pdf.read()
        base_name = Path(pdf.filename).stem if pdf.filename else "document"
        data_dir = Path(__file__).parent.parent / "data"
        t_pipeline = time.perf_counter()

        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)

        SURYA_DPI = 192
        CROP_DPI = 384  # high-res for readable crops
        crop_scale = CROP_DPI / SURYA_DPI

        t0 = time.perf_counter()
        surya_images = []
        hires_images = []
        surya_mat = fitz.Matrix(SURYA_DPI/72, SURYA_DPI/72)
        hires_mat = fitz.Matrix(CROP_DPI/72, CROP_DPI/72)
        for page_num in range(num_pages):
            pdf_page = doc[page_num]
            # 192 DPI for Surya
            pix = pdf_page.get_pixmap(matrix=surya_mat)
            surya_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            # 384 DPI for cropping
            pix = pdf_page.get_pixmap(matrix=hires_mat)
            hires_images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
        print(f"  [timing] PDF rendering ({num_pages} pages): {time.perf_counter()-t0:.2f}s")

        # Run Surya layout detection via Modal GPU
        t0 = time.perf_counter()
        layout_results = await asyncio.to_thread(detect_layout, surya_images)
        print(f"  [timing] Surya layout detection (Modal GPU): {time.perf_counter()-t0:.2f}s")

        # Detect figures missed by Surya (pixel-based gap detection)
        t0 = time.perf_counter()
        for page_idx, (img, layout_result) in enumerate(zip(surya_images, layout_results)):
            uncovered = detect_uncovered_figures(img, layout_result.bboxes)
            for bbox in uncovered:
                layout_result.bboxes.append(SyntheticBox(bbox=bbox, label="Picture"))
                print(f"  [reconstruct] Detected uncovered figure on page {page_idx+1}: {bbox}")

        # Annotate each page with continuous indexing
        annotated_pages = []
        current_index = 1
        for img, layout_result in zip(surya_images, layout_results):
            annotated, current_index = annotate_page(img, layout_result, scale=2, start_index=current_index)
            annotated_pages.append(annotated)

        total_annotations = current_index - 1

        # Convert annotated pages to JPEG bytes for Gemini
        page_images: list[bytes] = []
        for page in annotated_pages:
            buf = io.BytesIO()
            page.save(buf, format="JPEG", quality=85)
            page_images.append(buf.getvalue())
        print(f"  [timing] Figure detection + annotation + JPEG encode: {time.perf_counter()-t0:.2f}s")

        # Debug: save annotated pages as PDF
        if debug:
            annotations_dir = data_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            annotated_pages[0].save(
                annotations_dir / f"{base_name}.pdf",
                "PDF",
                save_all=True,
                append_images=annotated_pages[1:] if len(annotated_pages) > 1 else [],
                resolution=150,
            )

        # Create LLM client once for grouping, extraction, and compilation fix attempts
        from lib.llm_client import LLMClient
        llm_client = LLMClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
        )

        # Call Gemini to group annotations into problems
        t0 = time.perf_counter()
        prompt = GROUP_PROBLEMS_PROMPT.format(total_annotations=total_annotations)
        raw_response = await asyncio.to_thread(
            llm_client.generate,
            prompt=prompt,
            images=page_images,
            response_schema=GroupProblemsResponse.model_json_schema(),
        )
        group_result = GroupProblemsResponse.model_validate_json(raw_response)
        print(f"  [timing] Gemini problem grouping: {time.perf_counter()-t0:.2f}s")

        # Sort problems by first annotation index to preserve document order
        group_result.problems.sort(key=lambda p: min(p.annotation_indices) if p.annotation_indices else float('inf'))

        # Debug: save labels (problem groupings) as JSON
        if debug:
            labels_dir = data_dir / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            (labels_dir / f"{base_name}.json").write_text(
                group_result.model_dump_json(indent=2)
            )

        # Build bbox index: annotation_index -> (page_number, bbox, label)
        bbox_index: dict[int, tuple[int, list[float], str]] = {}
        ann_idx = 1
        for page_num, layout_result in enumerate(layout_results):
            for block in layout_result.bboxes:
                bbox_index[ann_idx] = (page_num, list(block.bbox), block.label)
                ann_idx += 1

        # Rescue orphaned figure annotations
        FIGURE_LABELS = {"Picture", "Figure"}
        assigned = set()
        for p in group_result.problems:
            assigned.update(p.annotation_indices)

        for idx, (page_num, bbox, label) in bbox_index.items():
            if idx not in assigned and label in FIGURE_LABELS:
                best_problem = min(
                    group_result.problems,
                    key=lambda p: min(abs(idx - i) for i in p.annotation_indices)
                )
                best_problem.annotation_indices.append(idx)
                best_problem.annotation_indices.sort()
                print(f"  [reconstruct] Rescued orphan figure {idx} ({label}) -> {best_problem.label}")

        # For each problem, send full annotated pages to the LLM for extraction.
        # Group problems by annotation indices to deduplicate LLM calls —
        # when multiple problems share the same annotations (common when Surya
        # detects one large text block), extract all of them in a single call.

        def _get_extraction_images(problem: ProblemGroup):
            """Get full annotated page images and figure data for a problem."""
            image_data: dict[str, str] = {}
            figure_filenames: list[str] = []
            figure_mappings: list[str] = []
            problem_pages: set[int] = set()

            for idx in problem.annotation_indices:
                if idx not in bbox_index:
                    continue
                page_num, bbox, label = bbox_index[idx]
                problem_pages.add(page_num)

                # Still extract figures from hi-res images
                if label in FIGURE_LABELS:
                    hires = hires_images[page_num]
                    x1 = max(0, int(bbox[0] * crop_scale))
                    y1 = max(0, int(bbox[1] * crop_scale))
                    x2 = min(hires.width, int(bbox[2] * crop_scale))
                    y2 = min(hires.height, int(bbox[3] * crop_scale))
                    if x2 > x1 and y2 > y1:
                        buf = io.BytesIO()
                        hires.crop((x1, y1, x2, y2)).save(buf, format="JPEG", quality=90)
                        fname = f"figure_{idx}.jpg"
                        image_data[fname] = base64.b64encode(buf.getvalue()).decode()
                        figure_filenames.append(fname)
                        figure_mappings.append(f"  - Red box #{idx} → {fname}")

            # Include page 0 as context if problem isn't on it
            # (page 1 typically has models, tables, reference data)
            if problem_pages and 0 not in problem_pages:
                problem_pages.add(0)

            # Crop tables from relevant pages as images
            # (tables may contain drawings/structural formulas that can't be
            # reproduced as LaTeX, so we embed the original image instead)
            for idx, (page_num, bbox, label) in bbox_index.items():
                if page_num in problem_pages and label == "Table":
                    hires = hires_images[page_num]
                    x1 = max(0, int(bbox[0] * crop_scale))
                    y1 = max(0, int(bbox[1] * crop_scale))
                    x2 = min(hires.width, int(bbox[2] * crop_scale))
                    y2 = min(hires.height, int(bbox[3] * crop_scale))
                    if x2 > x1 and y2 > y1:
                        buf = io.BytesIO()
                        hires.crop((x1, y1, x2, y2)).save(buf, format="JPEG", quality=90)
                        fname = f"table_{idx}.jpg"
                        image_data[fname] = base64.b64encode(buf.getvalue()).decode()
                        figure_filenames.append(fname)
                        figure_mappings.append(f"  - Table (Red box #{idx}) → {fname}")

            # Return full annotated pages in order
            extraction_images = [page_images[p] for p in sorted(problem_pages)]

            return extraction_images, image_data, figure_filenames, figure_mappings

        async def reconstruct_group(problems: list[ProblemGroup]) -> list[tuple[str, str, dict, dict | None]]:
            """Extract all questions sharing the same page regions in one LLM call."""
            extraction_images, image_data, figure_filenames, figure_mappings = _get_extraction_images(problems[0])

            labels_str = ", ".join(p.label for p in problems)
            print(f"  [reconstruct] Group [{labels_str}]: {len(extraction_images)} pages ({len(figure_filenames)} figures) from indices {problems[0].annotation_indices}")

            if not extraction_images:
                return [(p.label, "% No regions found", {}, None) for p in problems]

            # Build extraction prompt
            extract_prompt = EXTRACT_QUESTION_PROMPT
            if len(problems) == 1:
                extract_prompt += f"\n\n## Target Problem\nExtract ONLY **{problems[0].label}** from the annotated page images. The pages show numbered red bounding boxes — focus on the content within the relevant boxes. Other content on the page is context only."
            else:
                labels = [p.label for p in problems]
                nums = [re.findall(r"\d+", l)[0] for l in labels if re.findall(r"\d+", l)]
                extract_prompt += f"\n\n## Multiple Problems — CRITICAL\nThis image contains {len(labels)} SEPARATE numbered problems. Each one MUST be its own top-level Question object in the `questions` array.\n\nProblems to extract: {', '.join(labels)}\nExpected problem numbers: {', '.join(nums)}\n\nRules:\n- Return exactly {len(labels)} Question objects.\n- Each Question has its own `number` field matching the problem number shown in the document.\n- Do NOT nest different problem numbers as sub-parts of another question. Problem 9 is NOT a sub-part of Problem 8 — it is a separate question.\n- Only use `parts` for actual labeled sub-questions within a single problem (e.g. a, b, c)."

            if figure_filenames:
                extract_prompt += "\n\nFigure files available for this problem:\n" + "\n".join(figure_mappings)
                extract_prompt += "\n\nThese figures were detected adjacent to this problem. Include them in the `figures` list if the question needs them to be answerable — even if the question text doesn't explicitly say \"see figure.\""

            # Choose schema: single Question or QuestionBatch
            if len(problems) == 1:
                schema = Question.model_json_schema()
            else:
                schema = QuestionBatch.model_json_schema()

            raw = await asyncio.to_thread(
                llm_client.generate,
                prompt=extract_prompt,
                images=extraction_images,
                response_schema=schema,
            )

            if len(problems) == 1:
                questions = [Question.model_validate_json(raw)]
            else:
                batch = QuestionBatch.model_validate_json(raw)
                questions = batch.questions

            # Strip hallucinated figure filenames
            valid_figs = set(figure_filenames)
            for q in questions:
                q.figures = [f for f in q.figures if f in valid_figs]
                for part in q.parts:
                    part.figures = [f for f in part.figures if f in valid_figs]
                    for sub in part.parts:
                        sub.figures = [f for f in sub.figures if f in valid_figs]

            # Match extracted questions to problems by number
            q_by_number: dict[int, list[Question]] = defaultdict(list)
            for q in questions:
                q_by_number[q.number].append(q)

            out: list[tuple[str, str, dict, dict | None]] = []
            for problem in problems:
                nums = re.findall(r"\d+", problem.label)
                matched = None
                if nums:
                    target = int(nums[0])
                    candidates = q_by_number.get(target, [])
                    if candidates:
                        matched = candidates.pop(0)

                if matched is None:
                    # Fallback: take any remaining question
                    for remaining in q_by_number.values():
                        if remaining:
                            matched = remaining.pop(0)
                            break

                if matched:
                    latex = question_to_latex(matched)
                    print(f"  [reconstruct] {problem.label}: got {len(latex)} chars of LaTeX (structured)")
                    out.append((problem.label, latex, image_data, matched.model_dump()))
                else:
                    print(f"  [reconstruct] {problem.label}: no matching question in batch extraction")
                    out.append((problem.label, "% Extraction failed", {}, None))

            return out

        # Group problems by annotation indices (same indices = same crop)
        crop_groups: dict[tuple, list[ProblemGroup]] = defaultdict(list)
        for p in group_result.problems:
            key = tuple(sorted(p.annotation_indices))
            crop_groups[key].append(p)

        # Run one extraction per unique crop group in parallel
        t0 = time.perf_counter()
        group_tasks = [reconstruct_group(probs) for probs in crop_groups.values()]
        group_results_nested = await asyncio.gather(*group_tasks)
        print(f"  [timing] Gemini question extraction ({len(group_tasks)} groups): {time.perf_counter()-t0:.2f}s")

        # Flatten and reorder to match original problem order
        results_by_label: dict[str, tuple] = {}
        for group_list in group_results_nested:
            for r in group_list:
                results_by_label[r[0]] = r
        results = [results_by_label[p.label] for p in group_result.problems]

        # Save structured questions (always available now)
        questions = [q for _, _, _, q in results if q is not None]
        if questions:
            structured_dir = data_dir / "structured"
            structured_dir.mkdir(parents=True, exist_ok=True)
            (structured_dir / f"{base_name}.json").write_text(
                json.dumps(questions, indent=2)
            )
            print(f"  [extract] Saved {len(questions)} structured questions to data/structured/{base_name}.json")

        # Compile each LaTeX result to PDF in parallel
        from lib.latex_compiler import LaTeXCompiler
        compiler = LaTeXCompiler()

        async def compile_problem(problem_num: int, label: str, latex: str, image_data: dict[str, str], question_dict: dict | None) -> tuple[str, bytes | None, dict | None]:
            """Compile a single problem's LaTeX to PDF. On failure, ask LLM to fix the LaTeX and retry once."""
            header = f"Problem {problem_num}"
            content = f"\\textbf{{\\large {header}}}\n\n{latex}"
            print(f"  [compile] {label}: {len(latex)} chars, images={list(image_data.keys()) or 'none'}")
            try:
                pdf_bytes = await asyncio.to_thread(
                    compiler.compile_latex, content, image_data=image_data or None
                )
                return (label, pdf_bytes, question_dict)
            except Exception as e:
                print(f"  [compile] {label}: FAILED — {e}")
                # Try LLM fix
                try:
                    fix_prompt = LATEX_FIX_PROMPT.format(
                        latex_body=latex,
                        error_message=str(e)[:2000]
                    )
                    fixed_latex = await asyncio.to_thread(
                        llm_client.generate, prompt=fix_prompt
                    )
                    fixed_content = f"\\textbf{{\\large {header}}}\n\n{fixed_latex}"
                    pdf_bytes = await asyncio.to_thread(
                        compiler.compile_latex, fixed_content, image_data=image_data or None
                    )
                    print(f"  [compile] {label}: FIXED by LLM")
                    return (label, pdf_bytes, question_dict)
                except Exception as e2:
                    print(f"  [compile] {label}: FIX FAILED — {e2}")
                    fallback = f"\\textbf{{\\large {header}}}\n\n\\textit{{LaTeX compilation failed for this problem.}}"
                    pdf_bytes = await asyncio.to_thread(compiler.compile_latex, fallback)
                    return (label, pdf_bytes, None)  # No regions for fallback

        t0 = time.perf_counter()
        compile_tasks = [compile_problem(i + 1, label, latex, image_data, q_dict) for i, (p, (label, latex, image_data, q_dict)) in enumerate(zip(group_result.problems, results))]
        compiled = await asyncio.gather(*compile_tasks)
        print(f"  [timing] LaTeX compilation ({len(compile_tasks)} problems): {time.perf_counter()-t0:.2f}s")

        if split:
            # Return individual problem PDFs as JSON with region metadata
            from lib.region_extractor import extract_question_regions
            problem_pdfs = []
            for i, (label, pdf_bytes, question_dict) in enumerate(compiled):
                entry = {
                    "number": i + 1,
                    "label": label,
                    "pdf_base64": base64.b64encode(pdf_bytes).decode(),
                }
                if question_dict is not None:
                    region_data = extract_question_regions(pdf_bytes, question_dict)
                    entry["page_heights"] = region_data["page_heights"]
                    entry["regions"] = region_data["regions"]
                else:
                    entry["page_heights"] = []
                    entry["regions"] = []
                problem_pdfs.append(entry)
            print(f"  [timing] TOTAL pipeline: {time.perf_counter()-t_pipeline:.2f}s")
            return JSONResponse({
                "problems": problem_pdfs,
                "total_problems": len(problem_pdfs),
                "page_count": num_pages
            })

        # Merge all per-problem PDFs into one
        merged = fitz.open()
        for label, problem_pdf_bytes, _ in compiled:
            sub_doc = fitz.open(stream=problem_pdf_bytes, filetype="pdf")
            merged.insert_pdf(sub_doc)
            sub_doc.close()

        merged_bytes = merged.tobytes()
        merged.close()

        # Debug: save reconstructed PDF
        output_filename = f"{base_name}.pdf"
        if debug:
            reconstructions_dir = data_dir / "reconstructions"
            reconstructions_dir.mkdir(parents=True, exist_ok=True)
            output_path = reconstructions_dir / output_filename
            output_path.write_bytes(merged_bytes)

        # Return the merged PDF
        print(f"  [timing] TOTAL pipeline: {time.perf_counter()-t_pipeline:.2f}s")
        output = io.BytesIO(merged_bytes)

        return StreamingResponse(
            output,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={output_filename}",
                "X-Problem-Count": str(len(group_result.problems)),
                "X-Page-Count": str(num_pages),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


QUIZ_GENERATION_PROMPT = """\
You are generating a practice quiz for a student. Create {num_questions} questions on the topic: "{topic}".

Difficulty level: {difficulty}

Question types to include: {question_types}

{context_section}

{general_knowledge_section}

{additional_notes_section}

## Rules
- Each question must be valid LaTeX body content (no preamble, no \\documentclass, no \\begin{{document}}).
- Use inline math with $...$ and display math with \\[...\\].
- Escape LaTeX special characters in prose: \\& not &, \\% not %, \\# not #.
- For "open ended" questions: ask the student to show their work and explain their reasoning.
- For "multiple choice" questions: provide 4 options labeled (A), (B), (C), (D) using an itemized list.
- For "fill in the blank" questions: use \\underline{{\\hspace{{3cm}}}} for blanks.
- Vary the questions to test different aspects of the topic.
- Questions should be appropriate for the specified difficulty level.
- Do NOT include answers or solutions — this is a practice quiz for the student to work through.

Return a JSON object matching the provided schema.
"""


class QuizQuestionGenerated(BaseModel):
    """A single generated quiz question from the LLM."""
    number: int
    text: str
    topic: str | None = None
    answer_space_cm: float = 5.0


class QuizQuestionsGenerated(BaseModel):
    """Batch of generated quiz questions."""
    questions: list[QuizQuestionGenerated]


@app.post("/ai/generate-quiz")
async def ai_generate_quiz(request_body: QuizGenerationRequest):
    """
    Generate a practice quiz from RAG context and configuration.

    Pipeline:
    1. Build prompt from topic + difficulty + RAG context
    2. Call Gemini for structured question generation (LaTeX body)
    3. Compile each question to PDF via LaTeXCompiler
    4. On failure, use LLM auto-fix (same pattern as reconstruct)
    5. Return array of { number, pdf_base64, topic }
    """
    try:
        # Build context section
        if request_body.rag_context.strip():
            context_section = f"""## Course Materials Context
The following is relevant content from the student's course notes. Base your questions on this material:

---
{request_body.rag_context}
---"""
        else:
            context_section = "## No course materials provided.\nGenerate questions based on general knowledge of the topic."

        # General knowledge section
        if request_body.use_general_knowledge:
            general_knowledge_section = "You may also draw on general knowledge beyond the provided notes to create more comprehensive questions."
        else:
            general_knowledge_section = "IMPORTANT: Only create questions that can be answered using the provided course materials. Do not require knowledge beyond what is in the notes."

        # Additional notes section
        if request_body.additional_notes:
            additional_notes_section = f"## Additional Instructions\n{request_body.additional_notes}"
        else:
            additional_notes_section = ""

        # Format question types
        question_types_str = ", ".join(t.replace("_", " ") for t in request_body.question_types)

        prompt = QUIZ_GENERATION_PROMPT.format(
            num_questions=request_body.num_questions,
            topic=request_body.topic,
            difficulty=request_body.difficulty,
            question_types=question_types_str,
            context_section=context_section,
            general_knowledge_section=general_knowledge_section,
            additional_notes_section=additional_notes_section,
        )

        # Create LLM client
        from lib.llm_client import LLMClient
        llm_client = LLMClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model="google/gemini-3-flash-preview",
            base_url="https://openrouter.ai/api/v1",
        )

        # Generate questions via LLM with structured output
        raw_response = await asyncio.to_thread(
            llm_client.generate,
            prompt=prompt,
            response_schema=QuizQuestionsGenerated.model_json_schema(),
        )

        generated = QuizQuestionsGenerated.model_validate_json(raw_response)

        # Compile each question to PDF
        from lib.latex_compiler import LaTeXCompiler
        compiler = LaTeXCompiler()

        async def compile_quiz_question(q: QuizQuestionGenerated) -> QuizQuestionResponse:
            """Compile a single quiz question to PDF, with LLM fix on failure."""
            latex_body = quiz_question_to_latex(q.number, q.text, q.answer_space_cm)
            try:
                pdf_bytes = await asyncio.to_thread(compiler.compile_latex, latex_body)
                return QuizQuestionResponse(
                    number=q.number,
                    pdf_base64=base64.b64encode(pdf_bytes).decode(),
                    topic=q.topic,
                )
            except Exception as e:
                print(f"  [quiz] Question {q.number}: compile FAILED — {e}")
                # Try LLM fix
                try:
                    fix_prompt = LATEX_FIX_PROMPT.format(
                        latex_body=latex_body,
                        error_message=str(e)[:2000],
                    )
                    fixed_latex = await asyncio.to_thread(llm_client.generate, prompt=fix_prompt)
                    fixed_body = f"\\textbf{{Question {q.number}}}\n\n{fixed_latex}"
                    pdf_bytes = await asyncio.to_thread(compiler.compile_latex, fixed_body)
                    print(f"  [quiz] Question {q.number}: FIXED by LLM")
                    return QuizQuestionResponse(
                        number=q.number,
                        pdf_base64=base64.b64encode(pdf_bytes).decode(),
                        topic=q.topic,
                    )
                except Exception as e2:
                    print(f"  [quiz] Question {q.number}: FIX FAILED — {e2}")
                    # Fallback: compile a placeholder
                    fallback = f"\\textbf{{Question {q.number}}}\n\n\\textit{{Question generation failed. Please try again.}}\n\n\\vspace{{5.0cm}}"
                    pdf_bytes = await asyncio.to_thread(compiler.compile_latex, fallback)
                    return QuizQuestionResponse(
                        number=q.number,
                        pdf_base64=base64.b64encode(pdf_bytes).decode(),
                        topic=q.topic,
                    )

        # Compile all questions in parallel
        compile_tasks = [compile_quiz_question(q) for q in generated.questions]
        results = await asyncio.gather(*compile_tasks)

        return [r.model_dump() for r in results]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
