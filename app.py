import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from notebook.rag_engine import RAGEngine


# --- Request/Response Models ---

class IngestRequest(BaseModel):
    pdf_directory: str = "data/pdf/"

class IngestResponse(BaseModel):
    status: str
    chunks_added: int

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    content: str
    source_file: str
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


# --- App Lifespan (initialize RAGEngine once on startup) ---

rag_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    print("Starting up — initializing RAG Engine...")
    rag_engine = RAGEngine()
    yield
    print("Shutting down.")


# --- FastAPI App ---

app = FastAPI(
    title="RAG Automation API",
    description="API for ingesting PDFs and querying with LLM-powered answers",
    version="1.0.0",
    lifespan=lifespan
)


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not os.path.exists(request.pdf_directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.pdf_directory}")
    try:
        chunks_added = rag_engine.ingest(request.pdf_directory)
        return IngestResponse(status="success", chunks_added=chunks_added)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        result = rag_engine.query(request.question)
        sources = [
            Source(
                content=doc["content"][:300],
                source_file=doc["metadata"].get("source_file", "unknown"),
                score=round(doc["similarity_score"], 4)
            )
            for doc in result["retrieved_docs"]
        ]
        return QueryResponse(
            question=result["query"],
            answer=result["answer"],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
