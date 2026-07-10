# app/api/routes.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.api.schemas import QueryRequest, QueryResponse, HealthResponse
from app.agents.router import agent
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from app.core.config import settings
from app.core.prompts import QA_PROMPT
from app.rag.retriever import retriever
from app.guardrails.input_guard import input_guard
import logging
import time
import asyncio

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    """
    Health check + model warmup.
    Render pings this every 30s — keeps the model warm.
    First ping after cold start loads the model.
    Subsequent pings return instantly.
    """
    from app.rag.embedder import embedder
    embedder.embed_query("warmup")  # pre-warms model on first hit
    return HealthResponse(status="ok", version="1.0.0")



@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main query endpoint.
    Takes a customer question, returns a grounded answer.
    
    This is synchronous for simplicity in Phase 1.
    Phase 2 adds async + streaming as a separate endpoint.
    """
    start = time.time()

    try:
        result = agent.run(request.query)
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error. Please try again."
        )

    latency = round(time.time() - start, 3)
    logger.info(f"Query processed in {latency}s | Route: {result['route']}")

    # Safely build sources — handle missing keys
    sources = []
    for s in result.get("sources", []):
        sources.append({
            "category": s.get("category", ""),
            "intent": s.get("intent", ""),
            "original_question": s.get("original_question", "")
        })

    return QueryResponse(
        answer=result["answer"],
        route=result["route"],
        sources=sources,
        escalate=result["escalate"],
        pii_detected=result["pii_detected"],
        is_refusal=result["is_refusal"]
    )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming endpoint — tokens appear word by word in the UI.
    Why streaming matters for UX:
    A 3-second response FEELS faster when the first word appears in 300ms.
    Users start reading immediately instead of staring at a spinner.
    This is why every major chat UI (ChatGPT, Claude) streams by default.
    """
    # Run input guard first
    guard_result = input_guard.validate(request.query)
    if not guard_result["allowed"]:
        async def blocked():
            yield "I cannot process that request."
        return StreamingResponse(blocked(), media_type="text/plain")

    cleaned_query = guard_result["cleaned_query"]

    # Retrieve context
    docs = retriever.retrieve(cleaned_query)
    if not docs:
        async def no_docs():
            yield "I don't have enough information to answer that. Let me connect you with a human agent."
        return StreamingResponse(no_docs(), media_type="text/plain")

    context = "\n\n---\n\n".join([
        f"[Source {i+1}]\n{doc['document']}"
        for i, doc in enumerate(docs)
    ])
    prompt = QA_PROMPT.format(context=context, question=cleaned_query)

    # Stream from Groq
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.PRIMARY_MODEL,
        temperature=0.1,
        streaming=True
    )

    async def token_generator():
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                yield chunk.content
                await asyncio.sleep(0)  # yield control to event loop

    return StreamingResponse(token_generator(), media_type="text/plain")