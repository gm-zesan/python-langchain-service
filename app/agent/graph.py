from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import classifier_node, security_gate_node, route_node

def build_graph():
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("security_gate", security_gate_node)
    workflow.add_node("route", route_node)
    
    # Define edges
    workflow.set_entry_point("classifier")
    workflow.add_edge("classifier", "security_gate")
    workflow.add_edge("security_gate", "route")
    workflow.add_edge("route", END)
    
    return workflow.compile()

router_graph = build_graph()
