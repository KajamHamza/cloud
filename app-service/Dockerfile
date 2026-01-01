# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (drivers for SQL Server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 build-essential unixodbc-dev && \
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg && \
    curl https://packages.microsoft.com/config/debian/12/prod.list | tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18

# Create a non-root user (Security Best Practice & HF Requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements first (for caching)
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose Hugging Face's standard port
EXPOSE 7860

# Run with Gunicorn
# Timeout set to 120s for model loading
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
