import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any

from .state import GraphState
from ..config import config

llm = ChatOpenAI(
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_BASE_URL,
    model=config.LLM_MODEL,
    temperature=0.0
)

# 1. Classifier Node
def classifier_node(state: GraphState) -> Dict[str, Any]:
    messages = state["messages"]
    latest_query = messages[-1]["body"] if messages else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the intent classifier. 
Determine the intent of the user's latest message.
The 'route' MUST be one of: "CHAT", "KNOWLEDGE", "ACTION", "ANALYTICS", "UNCERTAIN", "OOD".

- CHAT: General greetings, thanks, or small talk.
- KNOWLEDGE: Asking for information, FAQs, policies.
- ACTION: Any request to modify state, create order, update profile, delete data.
- ANALYTICS: Queries asking for business metrics, performance, cash-in, sales, dues.
- OOD: Completely out of domain.
- UNCERTAIN: Ambiguous queries.

Output ONLY valid JSON with keys: 'route', 'confidence', 'reason'"""),
        ("user", "{query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"query": latest_query})
    
    try:
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        
        parsed = json.loads(content)
        return {
            "intent": parsed.get("route", "UNCERTAIN"),
            "confidence": parsed.get("confidence", 0.5)
        }
    except Exception:
        return {
            "intent": "UNCERTAIN",
            "confidence": 0.0
        }

# 2. Security Gate Node
def security_gate_node(state: GraphState) -> Dict[str, Any]:
    intent = state.get("intent")
    
    # Deterministic policy: ACTION is deferred and mapped to blocked_mutation
    if intent == "ACTION":
        return {
            "security_status": "blocked_mutation",
            "intent": "UNCERTAIN"  # Override intent
        }
        
    return {
        "security_status": "allowed"
    }

# 3. Route Formatter Node
def route_node(state: GraphState) -> Dict[str, Any]:
    return {
        "final_route": state.get("intent", "UNCERTAIN")
    }
