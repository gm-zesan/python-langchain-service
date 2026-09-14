from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MessageItem(BaseModel):
    role: str
    body: str

class RoutingRequest(BaseModel):
    messages: List[MessageItem]
    workspace_id: int = 1
    customer_id: str = "guest"

class RoutingResponse(BaseModel):
    route: str
    confidence: float
    security_status: str
    ambiguity_type: Optional[str] = None
    clarification_options: Optional[List[Dict[str, Any]]] = None
