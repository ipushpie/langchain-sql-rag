# Docker Setup Guide

## Prerequisites

- Docker and Docker Compose installed on your system
- Git (to clone the repository)

## Quick Start

1. Clone the repository (if not already done):
```bash
git clone <your-repo-url>
cd langchain-ragflow
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit `.env` and fill in keys like GOOGLE_API_KEY, OLLAMA_BASE_URL, etc.
```

3. Build and run with Docker Compose:
```bash
# Build and start the main FastAPI service
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

4. Access the application:

- FastAPI app: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Services Included

### langchain-ragflow (Main Application)
- Service name: `langchain-ragflow` (defined in `docker-compose.yml`)
- Container name: `langchain-ragflow-app`
- Port mapping: `8000:8000` (host:container)
- Description: FastAPI application that mounts chat routes under `/api/chat`.
- Health check: `http://localhost:8000/docs` (configured in compose)

## Environment Variables

These variables are referenced by `docker-compose.yml` and the app. Place them in a `.env` file or export in your environment.

### Common variables
- `GOOGLE_API_KEY` — Google API key for Gemini / Generative AI integrations
- `GOOGLE_GEMINI_MODEL_NAME` — Gemini model name to use (if applicable)
- `NODE_DATABASE_URL` / `DD_DATABASE_URL` — optional database connection strings used by parts of the app
- `OLLAMA_BASE_URL` — Base URL for Ollama (default: `http://localhost:11435`)
- `OLLAMA_MODEL_NAME` — Ollama model name

## Docker Commands

### Development
```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs langchain-ragflow

# Follow logs (stream)
docker-compose logs -f langchain-ragflow
```

## Healthchecks & Troubleshooting

- Check service status:
```bash
docker-compose ps
```

- Check the app health endpoint (inside the container or from host):
```bash
docker-compose exec langchain-ragflow curl -f http://localhost:8000/docs
```

## Reset Everything
```bash
# Stop and remove everything
docker-compose down -v --remove-orphans

# Remove all images
docker-compose down --rmi all

# Clean up Docker system
docker system prune -a
```

## GPU Support (Optional)

If you run Ollama locally and want GPU support, adapt the `docker-compose.yml` to add GPU device reservation for the Ollama service and ensure your host has the NVIDIA runtime configured.

Prerequisites for GPU support:
- NVIDIA GPU
- NVIDIA Docker runtime installed
- Docker Compose version that supports device reservations