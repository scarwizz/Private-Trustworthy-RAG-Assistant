# src/api/endpoints.py
"""
API endpoint definitions for the RAG service.
Defines routes for ingestion, querying, and health checks.
"""
import logging
import time
import uuid
from pathlib import Path
import tempfile
import shutil
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from .schemas import (
    QueryRequest,
    QueryResponse,
    IngestResponse,
    HealthResponse,
    ErrorResponse,
    SourceCitation
)
from .dependencies import get_rag_service, RAGService

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["rag"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest documents into the knowledge base",
    response_description="Number of documents processed and chunks created"
)
async def ingest_documents(
    files: List[UploadFile] = File(
        ...,
        description="List of files to ingest (PDF, TXT, MD, etc.)"
    ),
    service: RAGService = Depends(get_rag_service)
):
    """
    Accept file uploads and process them into the vector store.

    Supported formats: PDF, TXT, MD, and other text-based formats.
    Files are temporarily saved, processed, and then cleaned up.
    """
    start_time = time.time()

    # Validate file types
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    temp_dir = None

    try:
        # Create temporary directory for uploaded files
        temp_dir = Path(tempfile.mkdtemp(prefix="rag_ingest_"))
        file_paths = []

        for upload_file in files:
            # Validate file extension
            file_ext = Path(upload_file.filename).suffix.lower()
            if file_ext not in allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
                )

            # Save uploaded file to temp directory
            file_path = temp_dir / upload_file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)

            file_paths.append(str(file_path))
            logger.info(f"Saved uploaded file: {upload_file.filename}")

        # Process files through the RAG service
        num_chunks = await service.add_documents(file_paths)

        processing_time = (time.time() - start_time) * 1000

        return IngestResponse(
            chunks_processed=num_chunks,
            message=f"Successfully processed {len(files)} file(s) into {num_chunks} chunks",
            processing_time_ms=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except ValueError as e:
        # Handle validation errors from the service (e.g., no text extracted)
        logger.warning(f"Validation error during ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during ingestion: {str(e)}"
        )
    finally:
        # Clean up temporary directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the RAG system",
    response_description="Generated answer with source citations"
)
async def query_knowledge_base(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Process a natural language query and return an answer with supporting sources.

    Uses the RAG pipeline to retrieve relevant context and generate a response.
    """
    start_time = time.time()

    try:
        # Process query through the RAG service
        result = await service.answer_query(
            query=request.query,
            top_k=request.top_k
        )

        # Ensure we have a query ID
        if "query_id" not in result:
            result["query_id"] = str(uuid.uuid4())

        # Calculate processing time
        process_time_ms = (time.time() - start_time) * 1000
        result["processing_time_ms"] = round(process_time_ms, 2)

        # Format sources according to schema
        sources = []
        for src in result.get("sources", []):
            sources.append(SourceCitation(
                id=src.get("id", "unknown"),
                content=src.get("content", "")[:500],  # Ensure length limit
                metadata=src.get("metadata"),
                score=min(max(src.get("score", 0.0), 0.0), 1.0)  # Clamp to [0,1]
            ))

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
            query_id=result["query_id"],
            processing_time_ms=int(result["processing_time_ms"])
        )

    except ValueError as e:
        logger.warning(f"Validation error in query: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during query processing: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    response_description="Service health status"
)
async def health_check(service: RAGService = Depends(get_rag_service)):
    """
    Check the health of the service and its dependencies.

    Verifies that the RAG pipeline is initialized and responsive.
    """
    try:
        # Try to access the pipeline to verify it's initialized
        pipeline = service.pipeline

        # Perform a lightweight check (e.g., verify embedding model is loaded)
        # We'll do a simple embedding to ensure the model works
        _ = pipeline.embedder.embed_text("test")

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            services={
                "pipeline": "initialized",
                "embedding_model": "loaded",
                "vector_store": "connected"
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status="unhealthy",
                version="1.0.0",
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                services={
                    "error": str(e)
                }
            ).dict()
        )


# Include the router in the main app (will be done in main.py)