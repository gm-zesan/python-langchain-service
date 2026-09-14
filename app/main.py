from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from .models import RoutingRequest, RoutingResponse, ToolCallLog
from .agent.graph import router_graph

app = FastAPI(title="LangChain / LangGraph L2 Agentic Router Service")

@app.post("/route", response_model=RoutingResponse)
async def route_query(req: RoutingRequest):
    try:
        langchain_messages = []
        for m in req.messages:
            if m.role in ["user", "human"]:
                langchain_messages.append(HumanMessage(content=m.body))
            elif m.role in ["assistant", "ai"]:
                langchain_messages.append(AIMessage(content=m.body))

        initial_state = {
            "messages": langchain_messages,
            "raw_messages": [{"role": m.role, "body": m.body} for m in req.messages],
            "workspace_id": req.workspace_id,
            "customer_id": req.customer_id,
            "loop_count": 0,
            "tool_calls_log": [],
            "intent": None,
            "confidence": 0.0,
            "security_status": "allowed",
            "ambiguity_type": None,
            "clarification_options": None,
            "final_route": "UNCERTAIN",
            "final_answer": None
        }
        
        # Invoke LangGraph L2 Agentic Workflow
        result_state = router_graph.invoke(initial_state)
        
        tool_logs = [
            ToolCallLog(
                name=t.get("name", "unknown"),
                args=t.get("args", {}),
                result=t.get("result"),
                security_status=t.get("security_status", "allowed")
            )
            for t in result_state.get("tool_calls_log", [])
        ]
        
        return RoutingResponse(
            route=result_state.get("final_route", "UNCERTAIN"),
            confidence=result_state.get("confidence", 0.0),
            security_status=result_state.get("security_status", "allowed"),
            ambiguity_type=result_state.get("ambiguity_type"),
            clarification_options=result_state.get("clarification_options"),
            final_answer=result_state.get("final_answer"),
            tool_calls=tool_logs,
            iterations=result_state.get("loop_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
