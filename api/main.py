import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.core.rag_engine import RAGEngine
from src.agents.workflow import WorkflowManager
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Engineer Standard API", version="1.0.0")

# Dependency injection for services (Simplified for demo)
rag_service = RAGEngine(collection_name="production_docs")
workflow_manager = WorkflowManager()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Standard RAG Query Endpoint
    """
    try:
        result = await rag_service.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["source_documents"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/stream")
async def stream_workflow(request: QueryRequest):
    """
    Agentic Workflow with Streaming Response
    """
    async def event_generator():
        async for step in workflow_manager.run_workflow(request.query):
            # Format as Server-Sent Events (SSE)
            yield f"data: {step}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
