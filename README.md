# LangChain RAG / MCP FastAPI App

This repository contains a FastAPI application that integrates LangChain and related utilities for retrieval-augmented generation (RAG) and Model Context Protocol (MCP) workflows. The app exposes an API under the `/api/chat` prefix and is runnable both locally and via Docker Compose.

## What's in this repo
- `main.py` — FastAPI application entrypoint. It mounts the chat router at `/api/chat`.
- `app/` — Application code (routes, services, schemas, utils).
- `docker-compose.yml`, `Dockerfile` — container and orchestration configuration.
- `requirements.txt` — Python dependencies.

## Requirements
- Python 3.8+
- pip

## Quick install (local)
1. (Optional) create and activate a virtualenv:
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root (copy `.env.example` if present) and set required environment variables (see below).

## Run locally (development)
The FastAPI app is mounted in `main.py` and exposes routes under `/api/chat`. Start a development server with Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open the interactive docs at: http://localhost:8000/docs

## Docker / Docker Compose
This project includes a `docker-compose.yml` service named `langchain-ragflow` which builds the image from the repository `Dockerfile` and maps container port 8000 to host 8000. The compose service mounts `./app` into `/app/app` (read-only) for development convenience and includes a healthcheck against `/docs`.

Start the service with:
```bash
docker-compose up --build
```

Or run detached:
```bash
docker-compose up -d --build
```

## API surface
- Chat routes are available under `/api/chat` (see `app/routes/chat_routes.py`).
- OpenAPI docs: `/docs`

### Chat API Features
The chat API supports intelligent database querying with the following features:

- **Customer-specific filtering**: Include a `customer_id` in your request to filter results for a specific customer
- **Detailed responses**: Get comprehensive answers with specific data details, not just counts
- **Optimized prompts**: Faster processing with concise, efficient prompts
- **Multi-database support**: Automatically selects the appropriate database based on your question

### Example API Request
```json
{
  "question": "list all the data breaches",
  "navigation_routes": ["/security/breaches", "/compliance/incidents"],
  "customer_id": 123
}
```

### Example Response
```json
{
  "answer": "Found 5 data breaches:\n\n**Recent Breaches:**\n- **Security Incident Alpha** (Status: OPEN) - Occurred: 2025-06-05, Discovered: 2025-06-05\n- **Data Leak Beta** (Status: CLOSED) - Occurred: 2025-05-15, Discovered: 2025-05-16\n\nAll breaches require immediate attention for compliance review.",
  "routes": "/security/breaches"
}
```

## Environment variables
These are referenced by `docker-compose.yml` and the application. Set them in a `.env` file or export them in your shell.

- Required (for features you use):
  - `GOOGLE_API_KEY` — Google API key for Gemini / Google integrations.
  - `GOOGLE_GEMINI_MODEL_NAME` — Gemini model name to use (if using Google generative models).

- Optional / service configuration:
  - `DATABASE_URL` or `NODE_DATABASE_URL` / `DD_DATABASE_URL` — SQL database connection strings if you use DB features.
  - `OLLAMA_BASE_URL` — Base URL for Ollama (default: `http://localhost:11435`).
  - `OLLAMA_MODEL_NAME` — Ollama model name.

## Development notes
- The app includes a `chat` router (see `app/routes/chat_routes.py`) mounted at `/api/chat`.
- Use the interactive docs at `/docs` to explore request/response schemas.

## License
MIT
