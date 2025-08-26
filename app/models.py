from pydantic import BaseModel
from typing import Optional, Literal, List

class StartRequest(BaseModel):
    image_path: str

class ResumeRequest(BaseModel):
    thread_id: str
    status: Literal["approved", "needs_correction"]
    human_comment: Optional[str] = None

class GraphResponse(BaseModel):
    thread_id: str
    run_status: Literal["finished", "user_feedback"]
    assistant_response: Optional[str] = None
    item_id: Optional[int] = None

class ChatRequest(BaseModel):
    text_input: Optional[str] = None
    image_path: Optional[str] = None
    conversation_history: List[dict] = []
    user_email: Optional[str] = None
    user_name:Optional[str] = None
    thread_id: Optional[str] = None  # <-- optional for continuing session

class ChatResponse(BaseModel):
    response: str
    items: Optional[List[dict]] = None
    conversation_done: bool
    thread_id: str
