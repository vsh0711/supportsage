#building API layer 
from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # for conversation tracking

class Source(BaseModel):
    category: str
    intent: str
    original_question: str

class QueryResponse(BaseModel):
    answer: str
    route: str
    sources: list[Source]
    escalate: bool
    pii_detected: list[str]
    is_refusal: bool

class HealthResponse(BaseModel):
    status: str
    version: str