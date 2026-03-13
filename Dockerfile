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
# Environment variables (baked into image)
# =======================================================
ENV COMMON_DB_HOST=aws-1-ap-southeast-1.pooler.supabase.com
ENV COMMON_DB_PORT=6543
ENV COMMON_DB_USER=postgres.wjrxtspjbnuqtatgupym
ENV COMMON_DB_PASSWORD="#IGDIwebapp"
ENV COMMON_DB_SSLMODE=require
ENV SECRET_KEY=cama_integ
ENV JWT_ALGORITHM=HS256
ENV ENVIRONMENT=production
ENV CORS_ORIGINS=*
ENV DATA_DIR=/data

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
