from app.schemas.chat_schemas import AIbotstreamRequest
from app.services.chat_services import ask_question
from app.config.logger import logger
from fastapi import APIRouter

router = APIRouter()

class AIBot:    
    @router.post("/", tags=["Call copilot"])
    async def call_api_bot(
            request_body : AIbotstreamRequest
        ):
        try:
            result = ask_question(
                request_body.question, 
                request_body.navigation_routes,
                request_body.customer_id
            )
            
            logger.info("✅ API request completed")
            return result.get('answer')
        except Exception as ex:
            logger.error(f"❌ API error: {ex}")
            raise
