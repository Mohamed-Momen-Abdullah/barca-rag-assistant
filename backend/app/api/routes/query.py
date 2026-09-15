import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import retrieval_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if retrieval_service.is_loaded else "vector store not loaded",
        vector_store_loaded=retrieval_service.is_loaded,
        total_chunks=retrieval_service.total_chunks if retrieval_service.is_loaded else None,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not retrieval_service.is_loaded:
        raise HTTPException(status_code=503, detail="Vector store is not loaded yet")

    try:
        result = generate_answer(request.question)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the client as a 500
        logger.exception("Error generating answer")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return QueryResponse(answer=result["answer"], sources=result["sources"])
