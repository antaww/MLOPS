FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (if needed later, extend this list)
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies based on pyproject.toml
RUN pip install --no-cache-dir \
	fastapi==0.128.2 \
	faster-whisper==1.2.1 \
	python-multipart==0.0.22 \
	tiktoken>=0.12.0 \
	torch==2.10.0 \
	transformers==5.1.0 \
	uvicorn==0.40.0

# Copy application code
COPY app ./app
COPY src ./src
COPY main.py ./main.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

