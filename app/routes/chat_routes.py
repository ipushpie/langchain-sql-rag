from app.schemas.chat_schemas import AIbotstreamRequest
from app.services.chat_services import ask_question
from app.config.logger import get_logger
from fastapi import APIRouter

# Initialize logger
logger = get_logger(__name__)

router = APIRouter()

class AIBot:    
    @router.post("/", tags=["Call copilot"])
    async def call_api_bot(
            request_body : AIbotstreamRequest
        ):
        try:
            logger.info("Processing chat API request")
            logger.debug(f"Request details - Question: {request_body.question[:100]}{'...' if len(request_body.question) > 100 else ''}, Customer ID: {request_body.customer_id}")
            
            result = ask_question(
                request_body.question, 
                request_body.navigation_routes,
                request_body.customer_id
            )
            
            logger.info("Chat API request processed successfully")
            return result.get('answer')
        except Exception as ex:
            logger.error(f"Error in chat API: {ex}")
            raise
