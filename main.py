from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat_routes import router as chat_router
from app.config.logger import get_logger

# Initialize logging
logger = get_logger(__name__)

app = FastAPI(
    title="LangChain RAG Chat API",
    description="AI-powered chat API for database querying with RAG capabilities",
    version="1.0.0"
)

logger.info("Starting LangChain RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS middleware configured")

app.include_router(chat_router, prefix="/api/chat")

logger.info("Chat router mounted at /api/chat")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutdown initiated")