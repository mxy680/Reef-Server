FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including tectonic for LaTeX compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgraphite2-3 \
    libharfbuzz0b \
    libfontconfig1 \
    libfreetype6 \
    libssl3 \
    && curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/ \
    && rm -rf /var/lib/apt/lists/*

# Pre-warm tectonic bundle cache with ALL packages used in LATEX_TEMPLATE
# This prevents parallel runtime compiles from racing to download packages
RUN printf '\\documentclass[12pt,letterpaper]{article}\n\
\\usepackage[margin=1in]{geometry}\n\
\\usepackage{amsmath}\n\
\\usepackage{amssymb}\n\
\\usepackage{amsfonts}\n\
\\usepackage{graphicx}\n\
\\usepackage{booktabs}\n\
\\usepackage{array}\n\
\\usepackage{xcolor}\n\
\\usepackage{needspace}\n\
\\usepackage{algorithm}\n\
\\usepackage{algorithmic}\n\
\\usepackage{listings}\n\
\\usepackage{caption}\n\
\\usepackage{changepage}\n\
\\usepackage{lmodern}\n\
\\usepackage[T1]{fontenc}\n\
\\begin{document}\n\
$x^2 + y^2 = z^2$\n\
\\begin{tabular}{ll}\\toprule a & b \\\\\\bottomrule\\end{tabular}\n\
\\end{document}\n' > /tmp/warmup.tex \
    && tectonic /tmp/warmup.tex --outdir /tmp \
    && rm /tmp/warmup.tex /tmp/warmup.pdf

# Install CPU-only PyTorch first (much smaller)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application (Railway sets PORT env var)
# --timeout-keep-alive: Increase timeout for long-running question extraction
CMD uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 180
