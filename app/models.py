from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class MessageItem(BaseModel):
    role: str
    body: str

class RoutingRequest(BaseModel):
    messages: List[MessageItem]
    workspace_id: int = 1
    customer_id: str = "guest"

class ToolCallLog(BaseModel):
    name: str
    args: Dict[str, Any]
    result: Optional[str] = None
    security_status: str = "allowed"

class RoutingResponse(BaseModel):
    route: str
    confidence: float
    security_status: str
    ambiguity_type: Optional[str] = None
    clarification_options: Optional[List[Dict[str, Any]]] = None
    # L2 Agent Execution Metadata
    final_answer: Optional[str] = None
    tool_calls: List[ToolCallLog] = []
    iterations: int = 0
