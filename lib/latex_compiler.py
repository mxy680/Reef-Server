"""
LaTeX compilation service using tectonic.

Compiles LaTeX content to PDF for individual questions.
"""

import os
import subprocess
import tempfile
import base64
import shutil
from pathlib import Path
from typing import Optional


# LaTeX document template with essential packages
LATEX_TEMPLATE = r"""
\documentclass[12pt,letterpaper]{{article}}

% Page setup - generous margins to prevent overflow
\usepackage[margin=1in]{{geometry}}

% Math packages
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{amsfonts}}

% Graphics
\usepackage{{graphicx}}
\graphicspath{{{{{image_path}}}}}

% Tables
\usepackage{{booktabs}}
\usepackage{{array}}

% Colors
\usepackage{{xcolor}}

% Prevent page breaks in middle of sub-questions
\usepackage{{needspace}}

% Algorithm/pseudocode support
\usepackage{{algorithm}}
\usepackage{{algorithmic}}

% Code listings
\usepackage{{listings}}
\lstset{{basicstyle=\ttfamily\small, columns=fullflexible, breaklines=true}}

% Captions outside floats
\usepackage{{caption}}

% Indentation for nested sub-parts
\usepackage{{changepage}}

% Font improvements
\usepackage{{lmodern}}
\usepackage[T1]{{fontenc}}

% Prevent paragraph indentation
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{1em}}

% Remove page numbers for single-question pages
\pagenumbering{{gobble}}

\begin{{document}}

{content}

\end{{document}}
"""


class LaTeXCompiler:
    """Compiles LaTeX content to PDF using tectonic."""

    def __init__(self, tectonic_path: Optional[str] = None):
        """
        Initialize the compiler.

        Args:
            tectonic_path: Path to tectonic binary, defaults to system PATH
        """
        self.tectonic_path = tectonic_path or "tectonic"

        # Verify tectonic is available
        try:
            result = subprocess.run(
                [self.tectonic_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"tectonic check failed: {result.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "tectonic not found. Install with: curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh"
            )

    def compile_latex(
        self,
        latex_content: str,
        image_data: Optional[dict[str, str]] = None,
        work_dir: Optional[str] = None
    ) -> bytes:
        """
        Compile LaTeX content to PDF.

        Args:
            latex_content: The LaTeX document body (without preamble)
            image_data: Dict of filename -> base64 encoded image data
            work_dir: Working directory for compilation (temp if not specified)

        Returns:
            PDF file contents as bytes
        """
        # Create working directory
        if work_dir:
            temp_dir = Path(work_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            cleanup = False
        else:
            temp_dir = Path(tempfile.mkdtemp())
            cleanup = True

        try:
            # Write base64-encoded images to working directory
            images_dir = temp_dir / "images"
            if image_data:
                images_dir.mkdir(exist_ok=True)
                for img_name, img_b64 in image_data.items():
                    img_bytes = base64.b64decode(img_b64)
                    img_path = images_dir / img_name
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

            # Build full document
            image_path_latex = str(images_dir) + "/" if image_data else "./"
            full_document = LATEX_TEMPLATE.format(
                image_path=image_path_latex,
                content=latex_content
            )

            # Write LaTeX file
            tex_file = temp_dir / "question.tex"
            with open(tex_file, "w", encoding="utf-8") as f:
                f.write(full_document)

            # Run tectonic
            result = subprocess.run(
                [
                    self.tectonic_path,
                    str(tex_file),
                    "--outdir", str(temp_dir),
                    "--keep-logs",  # Keep logs for debugging
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(temp_dir)
            )

            if result.returncode != 0:
                # Read log file if it exists
                log_file = temp_dir / "question.log"
                log_content = ""
                if log_file.exists():
                    log_content = log_file.read_text()[-2000:]  # Last 2000 chars

                raise RuntimeError(
                    f"LaTeX compilation failed:\n{result.stderr}\n\nLog:\n{log_content}"
                )

            # Read PDF output
            pdf_file = temp_dir / "question.pdf"
            if not pdf_file.exists():
                raise RuntimeError("PDF file was not generated")

            return pdf_file.read_bytes()

        finally:
            if cleanup and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

