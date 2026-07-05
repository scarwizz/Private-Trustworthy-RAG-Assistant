# src/api/schemas.py
"""
Pydantic V2 schemas for API request/response models.
Defines strict validation for all endpoints.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, conint, confloat, constr


# Ingestion Models
class IngestTextItem(BaseModel):
    """Single text document for ingestion."""
    text: constr(min_length=1, max_length=10_000_000) = Field(
        ...,
        description="Raw text content of the document",
        examples=["This is a sample document about machine learning."]
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata associated with the document",
        examples=[{"source": "user_upload", "category": "technical"}]
    )


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    texts: Optional[List[IngestTextItem]] = Field(
        default=None,
        description="List of text documents to ingest (alternative to file upload)"
    )


class IngestResponse(BaseModel):
    """Response model for document ingestion."""
    chunks_processed: int = Field(
        ...,
        ge=0,
        description="Number of text chunks created and stored"
    )
    message: str = Field(
        ...,
        description="Human-readable status message"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )


# Query Models
class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    query: constr(min_length=1, max_length=2000) = Field(
        ...,
        description="Natural language question to ask the system",
        examples=["What is the capital of France?"]
    )
    top_k: conint(ge=1, le=20) = Field(
        default=5,
        description="Number of relevant documents to retrieve for context"
    )


class SourceCitation(BaseModel):
    """Model for a source citation in the response."""
    id: str = Field(
        ...,
        description="Unique identifier for the source chunk"
    )
    content: str = Field(
        ...,
        max_length=500,
        description="Text content of the source (truncated for brevity)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata associated with the source"
    )
    score: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Relevance score of the source (0.0 to 1.0)"
    )


class QueryResponse(BaseModel):
    """Response model for a query."""
    answer: str = Field(
        ...,
        description="Generated answer to the question"
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="List of source citations supporting the answer"
    )
    query_id: str = Field(
        ...,
        description="Unique identifier for this query request"
    )
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )


# Health Models
class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(
        ...,
        description="Overall health status (healthy, degraded, unhealthy)"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of the health check"
    )
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual services/components"
    )


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(
        ...,
        description="Error type or exception_name",
        example="VALIDATION_ERROR"
    )
    detail: str = Field(
        ...,
        description="Human-readable error message"
    )
    error_code: str = Field(
        ...,
        description="Machine-readable error code",
        example="ERR_VALIDATION_001"
    )