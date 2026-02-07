"""Deterministic Question → LaTeX converter.

Converts a structured Question object into LaTeX body text (no preamble,
no \\documentclass, no \\begin{document}).  The output is ready to be
wrapped by the LaTeX compiler's template.
"""

import re

from lib.models.question import Part, Question

# Control characters (except \n \t) that are invalid in LaTeX
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Regex to split text into math and non-math regions.
# DOTALL is required so \[...\] blocks spanning multiple lines are matched.
_MATH_SPLIT_RE = re.compile(r'(\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\))', re.DOTALL)


def _fix_json_latex_escapes(text: str) -> str:
    r"""Restore LaTeX commands corrupted by JSON escape interpretation.

    When LLMs output LaTeX in JSON strings, backslash sequences like \text
    can be interpreted as JSON escapes (\t = tab, \n = newline, \b = backspace,
    \f = formfeed, \r = carriage return). This restores them to valid LaTeX.
    """
    # \t (tab) before LaTeX suffixes: \text, \textbf, \textit, \textrm, \times, etc.
    text = re.sub(r'\t(ext|imes|heta|au)', r'\\t\1', text)
    # \n (newline) is too common to fix broadly — only fix within math delimiters
    # where a newline before eq/abla etc. is clearly a corrupted LaTeX command.
    # These are handled by the control character stripping below instead.
    # \b (backspace) before LaTeX suffixes: \begin, \bf, \bar, \beta, \binom, etc.
    text = re.sub(r'\x08(egin|f[{ ]|ar|eta|inom|ig|oldsymbol|oxed)', r'\\b\1', text)
    # \f (formfeed) before LaTeX suffixes: \frac, \forall, etc.
    text = re.sub(r'\x0c(rac|orall)', r'\\f\1', text)
    # \r (carriage return) before LaTeX suffixes: \rightarrow, \right, \rangle, etc.
    text = re.sub(r'\r(ight|angle|aise|enewcommand)', r'\\r\1', text)
    # \0 (null byte) before any LaTeX command suffix — restore to backslash
    text = re.sub(r'\x00([a-zA-Z])', r'\\\1', text)
    return text


def _sanitize_text(text: str) -> str:
    """Fix text issues caused by JSON serialization, not LLM errors.

    The LLM is instructed to output valid LaTeX. This function only fixes
    transport-layer corruption (JSON escape sequences mangling backslashes).
    """
    text = _fix_json_latex_escapes(text)
    text = _CONTROL_CHARS.sub('', text)
    return text


def question_to_latex(question: Question) -> str:
    """Convert a Question to a LaTeX body string.

    Element order at every level: text → figures → (parts | vspace).
    """
    lines: list[str] = []

    # Stem text
    if question.text:
        lines.append(_sanitize_text(question.text))
        lines.append("")

    # Stem figures
    if question.figures:
        lines.append(_render_figures(question.figures))
        lines.append("")

    # Parts or answer space
    if question.parts:
        for part in question.parts:
            lines.append(_render_part(part, depth=0))
            lines.append("")
    else:
        lines.append(f"\\vspace{{{question.answer_space_cm:.1f}cm}}")
        lines.append("")

    return "\n".join(lines).rstrip()


def _render_part(part: Part, depth: int) -> str:
    """Render a single part (and its subparts) to LaTeX."""
    lines: list[str] = []

    # Prevent page break between question text and answer space
    lines.append("\\needspace{4\\baselineskip}")
    # Label + text
    lines.append(f"\\textbf{{({part.label})}} {_sanitize_text(part.text)}")

    # Figures
    if part.figures:
        lines.append("")
        lines.append(_render_figures(part.figures))

    # Subparts or answer space
    if part.parts:
        lines.append("")
        for sub in part.parts:
            lines.append(_render_part(sub, depth=depth + 1))
            lines.append("")
    else:
        lines.append("")
        lines.append(f"\\vspace{{{part.answer_space_cm:.1f}cm}}")

    body = "\n".join(lines)

    # Indent nested parts (depth >= 1)
    if depth >= 1:
        body = f"\\begin{{adjustwidth}}{{1.5em}}{{0pt}}\n{body}\n\\end{{adjustwidth}}"

    return body


def _render_figures(filenames: list[str]) -> str:
    """Render one or more figures as LaTeX."""
    n = len(filenames)
    if n == 0:
        return ""

    if n == 1:
        return (
            "\\begin{center}\n"
            f"\\fbox{{\\includegraphics[width=0.45\\linewidth]{{{filenames[0]}}}}}\n"
            "\\end{center}"
        )

    # Side-by-side with minipages
    width = f"{0.93 / n:.2f}"
    parts: list[str] = []
    for fname in filenames:
        parts.append(
            f"\\begin{{minipage}}{{{width}\\linewidth}}\\centering\n"
            f"\\fbox{{\\includegraphics[width=\\linewidth]{{{fname}}}}}\n"
            f"\\end{{minipage}}"
        )
    return "\\begin{center}\n" + "\\hfill\n".join(parts) + "\n\\end{center}"
