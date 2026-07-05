# src/api/main.py
"""
Main FastAPI application entry point.
Configures the app with middleware, exception handlers, and routes.
"""
import logging
import time
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .endpoints import router as api_router
from .schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api.main")

# Create FastAPI app
app = FastAPI(
    title="Private Trustworthy RAG Assistant API",
    description="""\
Production-grade API for a local, privacy-preserving Retrieval-Augmented Generation system.

Features
- Document Ingestion: Accept PDF, TXT, MD files and process them into a vector store
- Intelligent Querying: Hybrid search with re-ranking and verifiable source citations
- Health Monitoring: Comprehensive health checks and metrics
- Privacy-First: All processing happens locally - no data leaves your environment

Architecture
- Backend: FastAPI with async endpoints
- Vector Store: ChromaDB with BGE-m3 embeddings
- LLM: Phi-3-mini or Qwen2-7B (GGUF quantized)
- Frontend: Streamlit client (separate service)
""",
    version="1.0.0",
    contact={
        "name": "API Support",
        "url": "https://github.com/your-org/private-trustworthy-rag-assistant",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://localhost:(3000|8501|8080)",  # Common dev ports
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with consistent error format."""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            detail=str(exc.detail),
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="VALIDATION_ERROR",
            detail="Invalid request parameters",
            error_code="VALIDATION_ERROR"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            detail="An unexpected error occurred",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and their processing time."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}ms"
    )
    return response

# Include API router
app.include_router(api_router)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Private Trustworthy RAG Assistant API",
        "version": "1.0.0",
        "docs": "/api/v1/docs",
        "health": "/api/v1/health"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Private Trustworthy RAG Assistant API...")
    # Initialize the RAG service singleton to warm up models
    from .dependencies import get_rag_service
    _ = get_rag_service()  # This will initialize the pipeline
    logger.info("API startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")
    # Any cleanup tasks would go here
    logger.info("API shutdown complete")