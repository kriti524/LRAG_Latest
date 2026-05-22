FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Flask port
EXPOSE 5001

# Streamlit port
EXPOSE 8501

# Start Flask app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"]