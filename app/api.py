from fastapi import APIRouter
from uuid import uuid4
from app.models import StartRequest, ResumeRequest, GraphResponse,ChatRequest,ChatResponse
from app.upload import graph
from app.search import run_search_flow
import json



router = APIRouter()


def run_graph_and_response(input_state, config):
    # Run the graph
    result = graph.invoke(input_state, config)
    state = graph.get_state(config)
    
    # Determine next node / run status
    next_nodes = getattr(state, "next", [])
    thread_id = config["configurable"].get("thread_id", str(uuid4()))
    run_status = "user_feedback" if "human_review" in next_nodes else "finished"

    # Return response safely
    item_id = state.values.get("item_id", None)
    return GraphResponse(
        thread_id=thread_id,
        run_status=run_status,
        assistant_response=result.get("assistant_response", ""),
        item_id=item_id
    )

@router.post("/lostfound/start", response_model=GraphResponse)
def start_verification(request: StartRequest):
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {"image_path": request.image_path}
    
    return run_graph_and_response(initial_state, config)

@router.post("/lostfound/resume", response_model=GraphResponse)
def resume_verification(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    state_update = {"status": request.status}
    if request.human_comment:
        state_update["human_comment"] = request.human_comment
    
    graph.update_state(config, state_update)
    
    return run_graph_and_response(None, config)

@router.post("/lostfound/search_chat", response_model=ChatResponse)
def search_chat(req: ChatRequest):
    # Use existing thread_id or generate a new one
    thread_id = getattr(req, "thread_id", None)
    # Run the search flow
    result = run_search_flow(
        query=req.text_input,
        image_path=req.image_path,
        user_email=req.user_email,
        user_name=req.user_name,
        thread_id=thread_id
    )
   

    ai_messages = [m for m in result.get("messages", []) if m.__class__.__name__ == "AIMessage"]
    
    assistant_text = ai_messages[-1].content if ai_messages else ""
    tool_messages = [m for m in result.get("messages", []) if m.__class__.__name__ == "ToolMessage"]
    last_tool_message = tool_messages[-1] if tool_messages else None
    conversation_done = False
    if last_tool_message:
        try:
            tool_content = json.loads(last_tool_message.content)
            conversation_done = tool_content.get("conversation_done", False)
        except Exception:
            conversation_done = False
        
    return ChatResponse(
        response=assistant_text,          
        items=result.get("items", []),    
        conversation_done=conversation_done,
        thread_id=result.get("thread_id")
    )
