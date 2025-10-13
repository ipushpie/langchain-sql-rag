from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.chat_routes import router as chat_router
from app.config.logger import logger

app = FastAPI(
    title="LangChain RAG Chat API",
    description="AI-powered chat API for database querying with RAG capabilities",
    version="1.0.0"
)

logger.info("🚀 Starting LangChain RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api/chat")

@app.on_event("startup")
async def startup_event():
    logger.info("✅ API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🔄 API shutting down")