# =======================================================
# 1️⃣ FRONTEND BUILD STAGE
# =======================================================
FROM node:20 AS frontend-builder

# Set working directory
WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source code
COPY frontend/ .

# Build with root base path
RUN npm run build --base=/


# =======================================================
# 2️⃣ BACKEND STAGE
# =======================================================
FROM python:3.11-slim

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    gcc libpq-dev gdal-bin python3-gdal && \
    rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ .

# Copy built frontend output into static folder
COPY --from=frontend-builder /app/frontend/dist ./static

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
