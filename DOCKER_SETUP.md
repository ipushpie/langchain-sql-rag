# Docker Setup Guide

## Prerequisites

- Docker and Docker Compose installed on your system
- Git (to clone the repository)

## Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone <your-repo-url>
   cd langchain_sql_mcp
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` file and add your actual API keys and configuration values.

3. **Build and run with Docker Compose**:
   ```bash
   # Build and start all services
   docker-compose up --build

   # Or run in detached mode
   docker-compose up -d --build
   ```

4. **Access the application**:
   - FastAPI app: http://localhost:8000
   - API documentation: http://localhost:8000/docs

## Services Included

### 1. langchain-ragflow (Main Application)
- **Port**: 8000
- **Description**: Standalone FastAPI application with LangChain integration
- **Health Check**: Available at `/docs`

## Environment Variables

### Required Variables
- `GOOGLE_API_KEY`: Your Google API key for Gemini
- `GOOGLE_GEMINI_MODEL_NAME`: Gemini model name to use

### Optional Variables
- `NODE_DATABASE_URL`: Database connection string for Node database
- `DD_DATABASE_URL`: Database connection string for DD database
- `OLLAMA_MODEL_NAME`: Ollama model to use (if connecting to external Ollama)
- `OLLAMA_BASE_URL`: External Ollama service URL (default: http://localhost:11435)

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

# Follow logs
docker-compose logs -f
```

### Database Operations
```bash
# Note: This application doesn't use a database
# Database operations are not applicable
```

## Production Deployment

1. **Set production environment variables**:
   - Use strong passwords
   - Set proper database URLs
   - Configure production API keys

2. **Enable security features**:
   - Use HTTPS
   - Set up proper firewall rules
   - Configure authentication

3. **Scale services** (if needed):
   ```bash
   docker-compose up --scale langchain-ragflow=3
   ```

4. **Use production-ready images**:
   - Consider using specific version tags
   - Implement health checks
   - Set up monitoring

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in `docker-compose.yml` if needed
2. **Permission issues**: Ensure Docker has proper permissions
3. **Memory issues**: Increase Docker memory limits for Ollama
4. **Database connection**: Check database credentials and network connectivity

### Health Checks
```bash
# Check service health
docker-compose ps

# Check individual service
docker-compose exec langchain-ragflow curl http://localhost:8000/docs
```

### Reset Everything
```bash
# Stop and remove everything
docker-compose down -v --remove-orphans

# Remove all images
docker-compose down --rmi all

# Clean up Docker system
docker system prune -a
```

## GPU Support (Optional)

To enable GPU support for Ollama, uncomment the GPU configuration in `docker-compose.yml`:

```yaml
ollama:
  # ... other config
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Prerequisites for GPU support**:
- NVIDIA GPU
- NVIDIA Docker runtime installed
- Docker Compose version 1.19.0+