import os
import httpx
from typing import Dict, Any, Optional
from langchain_core.tools import tool

LARAVEL_API_BASE_URL = os.getenv("LARAVEL_API_BASE_URL", "http://127.0.0.1:8000/api/v1/internal")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://127.0.0.1:8002")

@tool
def analytics_query(query: str) -> str:
    """
    Execute read-only analytics queries (aggregations, sales totals, customer metrics, order counts).
    Pass only the analytical query string. Context/workspace will be deterministically injected by the runtime.
    Do NOT use for creating or modifying records.
    """
    # Note: runtime context (workspace_id) is injected by the security gate / executor node
    return query

@tool
def knowledge_search(query: str) -> str:
    """
    Search business knowledge base, documentation, FAQs, and product information.
    Pass only the search query string. Context/workspace will be deterministically injected by the runtime.
    """
    return query

@tool
def memory_search(query: str) -> str:
    """
    Search persistent customer and conversation memory for user profile and past context.
    Pass only the search query string. Context/workspace will be deterministically injected by the runtime.
    """
    return query

def execute_analytics_query(query: str, workspace_id: int) -> str:
    url = f"{LARAVEL_API_BASE_URL}/analytics"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"query": query, "workspace_id": workspace_id})
            if resp.status_code == 200:
                return str(resp.json().get("data", resp.json()))
            return f"Error: Analytics query failed with HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Error: Failed to connect to Analytics service: {str(e)}"

def execute_knowledge_search(query: str, workspace_id: int) -> str:
    url = f"{LARAVEL_API_BASE_URL}/knowledge"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"query": query, "workspace_id": workspace_id})
            if resp.status_code == 200:
                return str(resp.json().get("data", resp.json()))
            return f"Error: Knowledge search failed with HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Error: Failed to connect to Knowledge service: {str(e)}"

def execute_memory_search(query: str, workspace_id: int, customer_id: str) -> str:
    url = f"{MEMORY_SERVICE_URL}/v1/memory/search"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(url, json={"query": query, "workspace_id": workspace_id, "customer_id": customer_id})
            if resp.status_code == 200:
                return str(resp.json().get("data", resp.json()))
            return f"Error: Memory search failed with HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Error: Failed to connect to Memory service: {str(e)}"

ALLOWED_READ_ONLY_TOOLS = {
    "analytics_query": analytics_query,
    "knowledge_search": knowledge_search,
    "memory_search": memory_search,
}
