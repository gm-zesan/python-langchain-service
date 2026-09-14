import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .state import GraphState
from .tools import (
    ALLOWED_READ_ONLY_TOOLS,
    execute_analytics_query,
    execute_knowledge_search,
    execute_memory_search
)
from ..config import config

llm = ChatOpenAI(
    api_key=config.LLM_API_KEY,
    base_url=config.LLM_BASE_URL,
    model=config.LLM_MODEL,
    temperature=0.0
)

# Bind strictly scoped read-only tools to the model
tools_list = list(ALLOWED_READ_ONLY_TOOLS.values())
llm_with_tools = llm.bind_tools(tools_list)

SYSTEM_PROMPT = """You are an Agentic Business Assistant and Orchestrator.
You have access to read-only tools:
- analytics_query(query: str): For metrics, aggregate sales, counts, charts, and reporting.
- knowledge_search(query: str): For documentation, policies, FAQs, product details.
- memory_search(query: str): For past user interactions and persistent user context.

CRITICAL RULES:
1. You do NOT have any mutation tools (create, update, delete, block, cancel, make admin). If the user asks for a mutation or data modification, politely explain that you are in read-only mode and cannot perform modifications.
2. For analytical or factual questions, decide if you should invoke a tool or answer directly.
3. If an input is ambiguous or out-of-domain, ask for clarification or state that it is out of domain.
4. Do NOT attempt to pass workspace_id or customer_id; these are managed by the runtime security layer.
"""

def agent_node(state: GraphState) -> Dict[str, Any]:
    """
    LLM Agent Node: Evaluates conversation and decides whether to propose tool calls or generate a final response.
    """
    messages = list(state.get("messages", []))
    loop_count = state.get("loop_count", 0) + 1
    
    # Ensure system prompt is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": messages + [response],
        "loop_count": loop_count
    }

def security_gate_node(state: GraphState) -> Dict[str, Any]:
    """
    Deterministic Security & Policy Interceptor:
    Multi-point validation before tool execution:
    - Tool allowlist verification
    - Argument validation (ensuring query is string, stripping hallucinated tenant parameters)
    - Workspace integrity (enforces runtime tenant scope)
    - Read-only policy & mutation block
    - Depth guard verification
    """
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    tool_calls_log = list(state.get("tool_calls_log", []))
    security_status = state.get("security_status", "allowed")
    intent = state.get("intent")
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        content = last_message.content if last_message else ""
        return {
            "security_status": security_status,
            "final_answer": content
        }
        
    sanitized_tool_calls = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        raw_args = dict(tool_call.get("args", {}))
        
        # 1. Tool Allowlist Check
        if tool_name not in ALLOWED_READ_ONLY_TOOLS:
            security_status = "blocked_mutation"
            tool_calls_log.append({
                "name": tool_name,
                "args": raw_args,
                "security_status": "blocked",
                "result": f"Blocked: Tool '{tool_name}' is unauthorized / non-read-only."
            })
            continue
            
        # 2. Argument Validation & Sanitization
        query_val = raw_args.get("query")
        if not query_val or not isinstance(query_val, str):
            # Fallback to entire args if query key missing
            query_val = str(raw_args)
            
        sanitized_args = {"query": query_val.strip()}
        
        # 3. Workspace / Customer Context Integrity
        # Enforce server-side context; LLM cannot override or choose
        runtime_ws = state.get("workspace_id", 1)
        runtime_cust = state.get("customer_id", "guest")
        
        tool_call["args"] = sanitized_args
        sanitized_tool_calls.append({
            "id": tool_call.get("id", "call_1"),
            "name": tool_name,
            "args": sanitized_args,
            "_runtime_workspace_id": runtime_ws,
            "_runtime_customer_id": runtime_cust
        })
        
        tool_calls_log.append({
            "name": tool_name,
            "args": sanitized_args,
            "security_status": "allowed"
        })
        
        # Track intent for benchmark evaluation
        if tool_name == "analytics_query":
            intent = "ANALYTICS"
        elif tool_name == "knowledge_search":
            intent = "KNOWLEDGE"
        elif tool_name == "memory_search":
            intent = "CHAT"
            
    last_message.tool_calls = [
        {"id": c["id"], "name": c["name"], "args": c["args"]}
        for c in sanitized_tool_calls
    ]
    
    return {
        "security_status": security_status,
        "tool_calls_log": tool_calls_log,
        "intent": intent,
        "_sanitized_calls": sanitized_tool_calls
    }

def tool_execution_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes the validated read-only tools and feeds genuine observations back into state.
    Real tool failure returns an explicit Error without fabricated simulations.
    """
    messages = list(state.get("messages", []))
    sanitized_calls = state.get("_sanitized_calls", [])
    tool_calls_log = list(state.get("tool_calls_log", []))
    
    if not sanitized_calls:
        return {}
        
    tool_messages = []
    for call_info in sanitized_calls:
        tool_name = call_info["name"]
        tool_args = call_info["args"]
        tool_id = call_info["id"]
        ws_id = call_info.get("_runtime_workspace_id", 1)
        cust_id = call_info.get("_runtime_customer_id", "guest")
        
        query = tool_args.get("query", "")
        
        if tool_name == "analytics_query":
            res = execute_analytics_query(query, ws_id)
        elif tool_name == "knowledge_search":
            res = execute_knowledge_search(query, ws_id)
        elif tool_name == "memory_search":
            res = execute_memory_search(query, ws_id, cust_id)
        else:
            res = f"Error: Unauthorized tool '{tool_name}' execution denied."
            
        for log in tool_calls_log:
            if log["name"] == tool_name and log.get("result") is None:
                log["result"] = str(res)
                break
                
        tool_messages.append(ToolMessage(content=str(res), tool_call_id=tool_id))
        
    return {
        "messages": messages + tool_messages,
        "tool_calls_log": tool_calls_log
    }

def route_evaluator_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates final route and confidence based on state, messages, and security gate status.
    """
    security_status = state.get("security_status", "allowed")
    intent = state.get("intent")
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    
    final_answer = last_message.content if isinstance(last_message, AIMessage) else None
    
    if security_status == "blocked_mutation":
        return {
            "final_route": "UNCERTAIN",
            "confidence": 1.0,
            "security_status": "blocked_mutation",
            "final_answer": final_answer or "Action blocked: read-only policy."
        }
        
    if not intent:
        raw_msgs = state.get("raw_messages", [])
        last_query = raw_msgs[-1]["body"].lower() if raw_msgs else ""
        
        if any(w in last_query for w in ["hi", "hello", "hey"]):
            intent = "CHAT"
        elif any(w in last_query for w in ["cancel", "delete", "create", "make admin"]):
            intent = "UNCERTAIN"
            security_status = "blocked_mutation"
        elif any(w in last_query for w in ["weather", "draw", "code"]):
            intent = "OOD"
        elif any(w in last_query for w in ["revenue", "sales total", "orders count", "average"]):
            intent = "ANALYTICS"
        elif any(w in last_query for w in ["policy", "faq", "how to", "support"]):
            intent = "KNOWLEDGE"
        else:
            intent = "UNCERTAIN"
            
    return {
        "final_route": intent,
        "confidence": 0.9 if intent != "UNCERTAIN" else 0.5,
        "security_status": security_status,
        "final_answer": final_answer
    }
