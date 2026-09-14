from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage

from .state import GraphState
from .nodes import (
    agent_node,
    security_gate_node,
    tool_execution_node,
    route_evaluator_node
)

MAX_LOOP_DEPTH = 5

def should_continue(state: GraphState) -> str:
    """
    Conditional routing function:
    - If max loop depth reached -> finalize
    - If last message has tool calls and security gate allowed -> execute tools
    - Otherwise -> finalize route
    """
    messages = state.get("messages", [])
    loop_count = state.get("loop_count", 0)
    last_message = messages[-1] if messages else None
    
    if loop_count >= MAX_LOOP_DEPTH:
        return "route_evaluator"
        
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "security_gate"
        
    return "route_evaluator"

def after_security_gate(state: GraphState) -> str:
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    # If all tool calls were stripped/blocked or none left
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "route_evaluator"
        
    return "tool_execution"

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("security_gate", security_gate_node)
    workflow.add_node("tool_execution", tool_execution_node)
    workflow.add_node("route_evaluator", route_evaluator_node)
    
    # Define edges
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "security_gate": "security_gate",
            "route_evaluator": "route_evaluator"
        }
    )
    
    workflow.add_conditional_edges(
        "security_gate",
        after_security_gate,
        {
            "tool_execution": "tool_execution",
            "route_evaluator": "route_evaluator"
        }
    )
    
    workflow.add_edge("tool_execution", "agent")
    workflow.add_edge("route_evaluator", END)
    
    return workflow.compile()

router_graph = build_graph()
