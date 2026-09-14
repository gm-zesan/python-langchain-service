from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    # Inputs
    messages: List[Dict[str, str]]
    workspace_id: int
    customer_id: str
    
    # Graph processing state
    intent: Optional[str]
    confidence: float
    security_status: str # allowed, blocked, UNCERTAIN
    ambiguity_type: Optional[str]
    clarification_options: Optional[List[Dict[str, Any]]]
    
    # Output
    final_route: str
