from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    # Inputs & Runtime Context
    messages: List[BaseMessage]
    raw_messages: List[Dict[str, str]]
    workspace_id: int
    customer_id: str
    
    # State tracking
    loop_count: int
    tool_calls_log: List[Dict[str, Any]]
    
    # Classification / Security state
    intent: Optional[str]
    confidence: float
    security_status: str # allowed, blocked_mutation, blocked_scope
    ambiguity_type: Optional[str]
    clarification_options: Optional[List[Dict[str, Any]]]
    
    # Final Output
    final_route: str
    final_answer: Optional[str]
