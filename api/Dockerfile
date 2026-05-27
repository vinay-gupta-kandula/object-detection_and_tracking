# api/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for OpenCV
# Changed libgl1-mesa-glx to libgl1 for newer Debian compatibility
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We use the headless version of OpenCV for container environments
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    opencv-python-headless \
    numpy \
    tritonclient[grpc]

# Copy the API code
COPY . /app/api

# Set the working directory to where the main.py is located
WORKDIR /app/api

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]