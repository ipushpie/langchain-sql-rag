from typing import Optional
from pydantic import BaseModel

class AIbotstreamRequest(BaseModel):
    question: str
    # customer_id: Optional[int]
    # chat_id: Optional[str]
    # entity_id:Optional[int]=None
    # context: Optional[list]=None
    # db_key: Optional[str]=None