from fastapi import FastAPI, HTTPException
import time
from .models import RoutingRequest, RoutingResponse
from .agent.graph import router_graph

app = FastAPI(title="LangChain Router Service")

@app.post("/route", response_model=RoutingResponse)
async def route_query(req: RoutingRequest):
    try:
        initial_state = {
            "messages": [{"role": m.role, "body": m.body} for m in req.messages],
            "workspace_id": req.workspace_id,
            "customer_id": req.customer_id
        }
        
        # Invoke LangGraph
        result_state = router_graph.invoke(initial_state)
        
        return RoutingResponse(
            route=result_state.get("final_route", "UNCERTAIN"),
            confidence=result_state.get("confidence", 0.0),
            security_status=result_state.get("security_status", "allowed"),
            ambiguity_type=result_state.get("ambiguity_type"),
            clarification_options=result_state.get("clarification_options")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
