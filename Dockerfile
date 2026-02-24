FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install torch with CUDA support specifically
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Install NVIDIA runtime libraries for CUDA 12
RUN pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
