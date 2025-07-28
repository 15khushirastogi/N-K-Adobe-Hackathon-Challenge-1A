# Use Python base image with AMD64 architecture
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install system dependencies required for pdfplumber/pdfminer
RUN apt-get update && apt-get install -y \
    gcc \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_sm

# Command to run when container starts
CMD ["python", "process_pdfs.py"]
