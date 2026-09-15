from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.retrieval import retrieval_service
from app.utils.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load the vector store + embedding model ONCE, not per request.
    configure_logging()
    retrieval_service.load()
    yield
    # Shutdown: nothing to clean up -- chromadb's PersistentClient has no
    # explicit close(), and ollama calls are stateless HTTP requests.


app = FastAPI(
    title="Barça History RAG Assistant API",
    description="Retrieval-Augmented Generation API for FC Barcelona history, signings, and stats.",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)
