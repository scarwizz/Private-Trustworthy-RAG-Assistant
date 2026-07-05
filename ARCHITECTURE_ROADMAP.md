# Private-Trustworthy-RAG-Assistant: Architectural Revamp & Implementation Roadmap

## Current State Assessment
The project currently implements a functional local RAG system with:
- **UI**: Streamlit monolith (UI + backend tightly coupled)
- **Vector Store**: ChromaDB (with FAISS fallback) using BGE-base-en-v1.5 embeddings
- **Generation**: Flan-T5-base (250M parameter seq2seq model)
- **Orchestration**: Custom RAG pipeline with basic explainability (source snippets, confidence scores)
- **Configuration**: Centralized config via `src/utils/config.py`

While functional for a demo, the architecture lacks production-grade qualities: tight coupling limits scalability, the models are outdated for high-quality RAG, and trustworthiness mechanisms are rudimentary.

## Proposed High-Level Architecture
The following text-based diagram illustrates the proposed decoupled, microservices-oriented architecture:

```
+-------------------+       +-------------------+       +-------------------+
|   Streamlit UI    | <----> |   FastAPI Gateway | <----> |   RAG Microservices |
| (Presentation)    | HTTPS | (API Gateway/Auth) | gRPC/HTTP | (Core Business Logic) |
+-------------------+       +-------------------+       +-------------------+
                                   |            |                   |
                                   |            |                   |
                  +----------------+    +--------+--------+  +--------+--------+
                  |            |    |                 |  |                   |
          +-------+-------+  +-----+-----+  +-----------+-----------+  +----+-------+
          | Embedding Svc |  | Vector Store |  | LLM Service       |  | Query Proc Svc|
          | (BGE-m3)    |  | (ChromaDB)   |  | (Phi-3-mini GGUF) |  | (Rewriting)   |
          +-------+-------+  +-----+-----+  +-----------+-----------+  +----+-------+
                  |            |    |                 |  |                   |
          +-------+-------+  +-----+-----+  +-----------+-----------+  +----+-------+
          | HF Embedding  |  | Persistent  |  | GGUF Loader   |  | Rule/LLM      |
          | Model         |  | Volume      |  | (llama.cpp)   |  | Hybrid        |
          +---------------+  +-------------+  +---------------+  +---------------+
```

**Key Characteristics**:
1. **Decoupled Concerns**: UI (Streamlit) communicates solely via HTTP/JSON with the FastAPI gateway.
2. **Modular Backend**: Each core capability (embedding, storage, generation, query processing) is an independently scalable service.
3. **Model Modernization**: Uses state-of-the-art compact LLMs (Phi-3-mini, Qwen2) via efficient llama.cpp/GGUF inference.
4. **Hybrid Retrieval**: Combines dense vector search (embeddings) with sparse BM25 for robust recall.
5. **Trustworthiness Layer**: Integrated fact-checking, confidence calibration, and citation verification.
6. **Observable & Testable**: Built-in hooks for RAGAS/TruLens evaluation and Prometheus/Grafana monitoring.

## Prioritized Implementation Roadmap

### Phase 1: Decouple UI & Backend (Foundation for Production)
**Goal**: Separate presentation layer from business logic to enable independent scaling, testing, and deployment.

**Steps**:
1. **Create FastAPI Backend Service** (`src/api/`):
   - Define Pydantic models for request/response schemas (`api/schemas.py`)
   - Implement core endpoints:
     - `POST /ingest` (accept files, trigger chunking/embedding/storage)
     - `POST /query` (accept question, return answer + sources)
     - `GET /health` (liveness/readiness checks)
   - Extract current RAG pipeline logic into service-layer classes (`api/services/`)
   - Initialize models/vector store at startup (singleton pattern) to avoid reload overhead

2. **Refactor Streamlit UI** (`src/ui/app.py`):
   - Remove all direct imports of `src.core.*` and data processing logic
   - Implement API client module (`src/ui/api_client.py`) handling HTTP calls to backend
   - Retain only UI components: chat interface, file uploader, document manager, settings
   - Use `st.session_state` for chat history; fetch data via API calls

3. **Establish Contract**:
   - Define OpenAPI spec (`api/openapi.yaml`) documenting all endpoints
   - Add request/response validation and error handling middleware
   - Implement basic authentication (API key header) for future extensibility

**Why This Makes It Top-Notch**:
- Separation of concerns is a hallmark of production systems
- Enables independent scaling (e.g., scale GPU-heavy LLM service separately from CPU-heavy embedding service)
- Facilitates CI/CD: backend can be tested without launching Streamlit
- Creates clear interface contracts, reducing integration bugs
- Makes the system deployable as a microservice in container orchestration (K8s/Docker Swarm)

### Phase 2: Modernize Embedding & LLM Models (Quality Leap)
**Goal**: Replace outdated Flan-T5-base and BGE-base with cutting-edge compact models that fit 8-16GB RAM constraints while delivering superior generative and retrieval quality.

**Steps**:
1. **Embedding Upgrade**:
   - Replace `BAAI/bge-base-en-v1.5` with `BAAI/bge-m3` or `nomic-ai/nomic-embed-text-v1.5`
   - Both offer superior retrieval performance (especially bge-m3 with dense+sparse+colbert vectors) while staying ~560M parameters
   - Update `src/models/embeddings.py` to use new model (same interface)
   - Quantize to int8 if needed via `optimum` or `bitsandbytes` to reduce VRAM constraints`

2. **LLM Upgrade & Inference Optimization**:
   - Replace `google/flan-t5-base` with one of:
     - `microsoft/Phi-3-mini-4k-instruct` (3.8B params, ~4.5GB 4-bit)
     - `Qwen/Qwen2-7B-Instruct` (7B params, ~4.5GB 4-bit)
     - `mistralai/Mistral-7B-Instruct-v0.2` (7B params, ~4.5GB 4-bit)
   - Switch inference engine to `llama-cpp-python` (GGUF quantization) for CPU/GPU efficiency:
     - Convert model to GGUF format using `llama.cpp` conversion script
     - Create `src/models/llm_gguf.py` wrapper exposing `.generate()` method
   - Implement efficient batching and caching (LRU cache for recent embeddings/queries)

3. **Hardware Adaptation Layer**:
   - Abstract model loading behind a factory (`src/models/factory.py`) that selects:
     - GGUF/llama.cpp path if GPU with sufficient VRAM detected
     - Hugging Face `transformers` + `bitsandbytes` 4-bit quantization fallback
     - Pure CPU fallback (warning about latency)
   - Update config to specify model paths and quantization preferences

**Why This Makes It Top-Notch**:
- Uses SOTA compact LLMs that rival 70B+ models on reasoning benchmarks when properly quantized
- Demonstrates knowledge of modern LLM optimization techniques (quantization, efficient inference)
- Embedding upgrade (bge-m3) provides state-of-the-art retrieval accuracy (MTEB leaderboard)
- Shows ability to work within hardware constraints – critical for real-world deployments
- Positions the project as cutting-edge rather than academic toy

### Phase 3: Advanced RAG Architecture (Beyond Naive Vector Search)
**Goal**: Implement production-grade retrieval techniques that significantly improve relevance and reduce hallucination.

**Steps**:
1. **Hybrid Search (Dense + Sparse)**:
   - Implement BM25 indexing alongside dense vectors (use `whoosh` or `rank-bm25` pure-Python)
   - Store raw text in BM25 index during ingestion
   - At query time: retrieve top-k from dense vector store AND top-k from BM25
   - Fuse scores using Reciprocal Rank Fusion (RRF) or weighted sum
   - Create `src/retrieval/hybrid_searcher.py` module

2. **Re-Ranking with Cross-Encoder**:
   - Load lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
   - Take top-N (e.g., 50) from hybrid search, re-rank to get final top-K
   - Implement as `src/retrieval/reranker.py` with batching for efficiency
   - Cache reranker model to avoid reload latency

3. **Query Transformation & Routing**:
   - Implement query expansion: use LLM to generate 2-3 paraphrases or hypothetical answers (HyDE)
   - Implement simple query routing: detect if query is definitional, comparative, procedural etc. and boost relevant metadata filters
   - Create `src/query/transformer.py` module with pluggable strategies

4. **Contextual Chunking Upgrade**:
   - Replace fixed-size chunking with semantic chunking:
     - Use sentence embeddings to detect semantic shifts
     - Implement algorithm: embed sentences, compute cosine distance between adjacent, split when distance > threshold
   - Alternative: use layout-aware PDF parsing (if needed) with `pdfminer.s6` or `unstructured`
   - Update `src/data/chunker.py` to expose multiple strategies

**Why This Makes It Top-Notch**:
- Hybrid retrieval addresses the "vocabulary mismatch" problem of pure dense vectors
- Re-ranking significantly improves precision@k (critical for user trust)
- Query transformation shows understanding of advanced RAG techniques (HyDE, query routing)
- Semantic chunking preserves contextual coherence better than fixed-size splits
- Collectively, these techniques are what separate production RAG systems from academic demos

### Phase 4: Trustworthiness & Explainability (Core USP)
**Goal**: Implement measurable, programmable guarantees about output correctness and provenance.

**Steps**:
1. **Verifiable Source Citations**:
   - Enhance metadata stored with each chunk: include exact byte offsets in source document
   - At response generation, highlight exact span(s) used to support each claim
   - Implement in `src/explain/citation.py` – return character offsets` and UI highlighting

2. **Hallucination Detection & Confidence Scoring**:
   - Integrate natural language inference (NLI) model (e.g., `microsoft/deberta-large-mnli`)
   - For each sentence in generated response:
     - Premise = concatenated relevant source snippets
     - Hypothesis = the sentence
     - Get entailment/neutral/contradiction scores
   - Aggregate to get overall faithfulness score (0-1)
   - Implement fallback: if max entailment < threshold, trigger abstention or "I don't know"
   - Create `src/trust/hallucination_detector.py`

3. **Confidence Calibration**:
   - Use temperature scaling or isotonic regression on NLI scores to calibrate confidence
   - Display calibrated confidence alongside each citation in UI
   - Implement in `src/trust/calibration.py`

4. **Attention & Provenance Visualization** (Optional but impressive):
   - For transformer LLMs, extract attention weights linking generated tokens to source tokens
   - Requires custom GGUF build or using `transformers` with output attentions
   - Visualize in UI as heatmap over source text

**Why This Makes It Top-Notch**:
- Provides concrete, programmable mechanisms to detect and mitigate hallucinations
- Moves beyond "show sources" to actually verifying claims against sources
- Demonstrates deep understanding of Trustworthy AI principles (transparency, accountability)
- Directly addresses the #1 concern in enterprise LLM adoption: reliability
- Makes the system suitable for high-stakes domains (medical, legal, finance)

### Phase 5: Evaluation & Continuous Improvement (Metrics-Driven)
**Goal**: Establish an automated evaluation pipeline to continuously measure and improve RAG quality.

**Steps**:
1. **Create Evaluation Dataset**:
   - Curate 100-200 question-answer pairs from domain documents with human-verified ground truth
   - Store in `data/eval/` as JSONL: `{question, ground_truth, relevant_doc_ids}`

2. **Integrate RAGAS/TruLens**:
   - Add `ragas` and `trulens` to `requirements.txt`
   - Implement evaluation script (`scripts/evaluate_rag.py`) that:
     - Runs retrieval/generation on evaluation set
     - Computes RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall
     - Optionally uses TruLens for deep call-stack analysis
   - Schedule weekly runs via cron or GitHub Actions

3. **Feedback Loop**:
   - Add endpoint `POST /feedback` to collect user thumbs-up/down on answers
   - Store feedback with query, answer, retrieved contexts
   - Monthly review to identify failure patterns and update prompts/chunking/re-ranking thresholds

4. **Monitoring & Alerting**:
   - Instrument key metrics (latency, token usage, failure rate) with Prometheus
   - Create Grafana dashboard showing:
     - Query latency over time
     - Trustworthiness score distribution
     - Retrieval metrics (hit rate, MRR)
   - Set alerts for degradation (e.g., faithfulness < 0.8 for 5 consecutive runs)

**Why This Makes It Top-Notch**:
- Shifts from "it works on my machine" to data-driven quality assurance
- Demonstrates mastery of MLOps/LLMOps principles for LLM applications
- Provides concrete evidence of system reliability for resumes/interviews
- Enables continuous improvement – critical for production systems
- Shows ability to think beyond code to measurement and iteration

### Phase 5: Dockerization & Deployment (Instant Reproducibility)
**Goal**: Make the system trivially deployable with zero configuration frustration.

**Steps**:
1. **Multi-Stage Dockerfiles**:
   - `backend/Dockerfile`: 
     - Base: `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (if GPU) or `python:3.12-slim`
     - Install system deps: `ffmpeg`, `poppler-utils` (for PDF), `git`
     - Install Python deps via `poetry` or `pip` from `requirements.txt`
     - Copy model files (or mount volumes for models)
     - Expose port 8000
   - `frontend/Dockerfile`:
     - Base: `python:3.12-slim`
     - Install streamlit and dependencies
     - Copy Streamlit app
     - Expose port 8501

2. **docker-compose.yml**:
   ```yaml
   version: '3.8'
   services:
     backend:
       build: ./backend
       ports: ["8000:8000"]
       volumes:
         - ./models:/app/models:ro
         - ./data:/app/data
         - ./chroma_db:/app/chroma_db
       environment:
         - MODEL_PATH=/app/models
         - VECTOR_DB_PATH=/app/chroma_db
     frontend:
       build: ./frontend
       ports: ["8501:8501"]
       depends_on:
         - backend
       environment:
         - BACKEND_URL=http://backend:8000
   ```

3. **Documentation & Automation**:
   - Create `MAKEFILE` with targets: `build`, `up`, `down`, `logs`, `test`
   - Add GitHub Actions workflow to build and test Docker images on push
   - Create `docs/deployment.md` with step-by-step guides for:
     - Local development (`docker compose up`)
     - GPU-enabled deployment
     - Production deployment with NGINX reverse proxy and TLS

**Why This Makes It Top-Notch**:
- Eliminates "works on my machine" problems – critical for portfolio pieces
- Demonstrates DevOps awareness and containerization best practices
- Makes the system immediately usable by recruiters/hiring managers
- Shows understanding of production concerns: dependency isolation, resource management
- Enables easy benchmarking and comparison with other approaches

## Expected Outcomes & Resume Impact
Upon completing this roadmap, the system will transform from a basic demo to a portfolio-worthy showcase that demonstrates:

1. **Systems Engineering**: Microservices, API design, containerization
2. **ML Engineering**: Model quantization, efficient inference, embedding optimization
3. **Advanced RAG**: Hybrid search, re-ranking, query transformation
4. **Trustworthy AI**: Hallucination detection, verifiable citations, confidence calibration
5. **MLOps**: Continuous evaluation, feedback loops, production monitoring
6. **Full-Stack**: Modern Python packaging, type hints, testing, documentation

Each phase delivers standalone value and can be presented as distinct accomplishments on a resume. The final system will be ready for deployment in privacy-sensitive enterprise environments – exactly what employers seek in senior AI/ML engineers.

---
*Next Steps*: 
1. Review this roadmap with stakeholders
2. Phase 1: Begin by extracting the RAG pipeline into `api/services/rag_service.py`
3. Establish API contract before modifying UI
4. Proceed iteratively, validating each phase with the evaluation suite