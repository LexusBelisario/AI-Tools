# =======================================================
# 1) FRONTEND BUILD STAGE
# =======================================================
FROM node:20 AS frontend-builder

WORKDIR /app/frontend

# Install deps
COPY frontend/package*.json ./
RUN npm install

# Copy source then build
COPY frontend/ .

# Vite env vars must be set BEFORE build (they get baked in)
ENV VITE_API_URL=""
ENV VITE_TRUSTED_ORIGINS="http://localhost:5173,https://cama-core-14282293226.asia-southeast1.run.app,http://localhost:8000,http://35.194.255.28:8000"

RUN npm run build --base=/


# =======================================================
# 2) BACKEND STAGE
# =======================================================
FROM python:3.11-slim

WORKDIR /app

# System deps for geopandas/gdal/psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    gdal-bin \
    python3-gdal \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app code into /app
COPY backend/ .

# Put frontend build into backend static folder (matches main.py mount)
COPY --from=frontend-builder /app/frontend/dist ./static

# Data dir for docker-safe file paths
RUN mkdir -p /data

# =======================================================
# Environment variables (baked into image for partner deployment)
# Partner only needs: docker run -p 8003:8003 ai-tools
# =======================================================
ENV COMMON_DB_HOST=34.143.153.78
ENV COMMON_DB_PORT=5432
ENV COMMON_DB_USER=blgf_gis_user
ENV COMMON_DB_PASSWORD=Wo8sheiweedohhe2!
ENV COMMON_DB_SSLMODE=require
ENV SECRET_KEY=cama_integ
ENV JWT_ALGORITHM=HS256
ENV ENVIRONMENT=production
ENV DATA_DIR=/app/backend
ENV DB_POOL_SIZE=5
ENV DB_MAX_OVERFLOW=10
ENV CORS_ORIGINS=*

EXPOSE 8003

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]