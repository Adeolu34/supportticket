FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright and Chromium
RUN playwright install chromium && \
    playwright install-deps chromium

# Download NLTK data required by classifier_service
RUN python -c "\
import nltk; \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
"

# Copy application code
COPY --chown=appuser:appgroup . .

# Create directories
RUN mkdir -p /app/screenshots /app/models /app/logs && \
    chown -R appuser:appgroup /app/screenshots /app/models /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8004

# Default command (overridden by docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"]
