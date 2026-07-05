# src/ui/app.py
"""
Streamlit UI for the TAI local RAG assistant (Phase 1+).
- Chat interface (uses API client to communicate with FastAPI backend)
- Document management (upload --> save temporarily --> send to backend for ingestion)
- All core logic is handled by the backend services and only on API.

Notes:
- This code is intentionally defensive: it will try common API methods and
  fallback with clear errors and guidance if something isn't available.
- The UI assumes everything runs locally/offline.
- Use st.session_state for caching API client and chat state.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from streamlit import session_state as state

# --- Fix import path for Streamlit context ---
ROOT = Path(__file__).resolve().parents[2]  # D:\TAI
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import our API client
try:
    from ui.api_client import get_api_client
except ImportError as e:
    st.error(f"Failed to import API client: {e}")
    st.stop()

# Configuration
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

# UI constants
APP_TITLE = "TAI — Local RAG Assistant"

# ----------------------------
# Helpers & caching functions
# ----------------------------

def safe_filename(name: str) -> str:
    """Return safe filename for saved uploads."""
    return "".join(c for c in name if c.isalnum() or c in " ._-").strip()


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded file into DOCS_DIR and return Path.
    Overwrites same-named files intentionally (user action).
    """
    filename = safe_filename(uploaded_file.name)
    target = DOCS_DIR / filename
    with open(target, "wb") as f:
        f.write(uploaded_file.getvalue())
    return target


def ensure_session_state():
    """Ensure session state keys exist for chat and UI behavior."""
    if "chat_history" not in state:
        state.chat_history = []  # list of {"role": "user"/"assistant", "text": "...", "meta": {...}}
    if "last_query" not in state:
        state.last_query = ""
    if "doc_count" not in state:
        state.doc_count = None
    if "backend_url" not in state:
        state.backend_url = "http://localhost:8000"


# ----------------------------
# API interaction functions
# ----------------------------

def query_backend(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the backend API and return the response.
    """
    try:
        client = get_api_client()
        # Update base URL if changed in session state
        if state.backend_url:
            client.base_url = state.backend_url.rstrip("/") + "/api/v1"
        # Now client.query is synchronous
        result = client.query(query, top_k=top_k)
        return result
    except Exception as e:
        st.error(f"Failed to query backend: {e}")
        raise


def ingest_files_backend(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Ingest files via the backend API.
    """
    try:
        client = get_api_client()
        if state.backend_url:
            client.base_url = state.backend_url.rstrip("/") + "/api/v1"
        # Now client.ingest_files is synchronous
        result = client.ingest_files(file_paths)
        return result
    except Exception as e:
        st.error(f"Failed to ingest files: {e}")
        raise


def check_backend_health() -> Dict[str, Any]:
    """
    Check the health of the backend API.
    """
    try:
        client = get_api_client()
        if state.backend_url:
            client.base_url = state.backend_url.rstrip("/") + "/api/v1"
        # Now client.health_check is synchronous
        result = client.health_check()
        return result
    except Exception as e:
        st.warning(f"Backend health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


# ----------------------------
# Document management functions
# ----------------------------

def list_documents_from_backend() -> List[Dict[str, Any]]:
    """
    Try to get list of documents from backend.
    Since we don't have a list endpoint yet, we'll fallback to local docs dir.
    In a full implementation, the backend would have a /documents endpoint.
    """
    # For now, we'll just list files in the docs directory
    files = []
    for f in DOCS_DIR.glob("*"):
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({"name": f.name, "path": str(f)})
    return files


def add_documents_to_backend(file_paths: List[Path]) -> Dict[str, Any]:
    """
    Wrapper to call the ingest function from sync Streamlit.
    """
    try:
        result = ingest_files_backend(file_paths)
        return {"ok": True, "msg": result.get("message", "Documents ingested successfully"), **result}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def delete_document_from_backend(name_or_id: str) -> Dict[str, Any]:
    """
    Delete a document by name or id.
    Since we don't have a delete endpoint yet, we'll fallback to local file deletion
    and suggest index rebuild.
    """
    target = DOCS_DIR / name_or_id
    if target.exists():
        try:
            target.unlink()
            return {"ok": True, "msg": f"Deleted local file {name_or_id} from docs dir; index may need rebuild via backend."}
        except Exception as e:
            return {"ok": False, "msg": f"Failed to delete local file: {e}"}
    else:
        return {"ok": False, "msg": f"Could not find local file {name_or_id}."}


def rebuild_index_backend() -> Dict[str, Any]:
    """
    Trigger a reindex via backend (if endpoint exists) or fallback to local.
    For now, we'll just report that the backend doesn't have this endpoint.
    """
    # In a full implementation, we'd call a POST /reindex endpoint
    return {"ok": False, "msg": "Rebuild index not implemented in backend yet. Please restart backend to reindex."}


# ----------------------------
# Chat & query functions
# ----------------------------

def query_backend_and_render(query: str):
    """
    Send the query to backend API, append to session chat,
    and render answer with explainability elements.
    """
    if not query.strip():
        st.warning("Please enter a question.")
        return

    # Add user message to chat history immediately
    state.chat_history.append({"role": "user", "text": query})
    state.last_query = query

    # Show spinner while waiting for backend
    with st.spinner("Generating answer..."):
        try:
            resp = query_backend(query, top_k=5)
        except Exception as e:
            st.error(f"Failed to get answer from backend: {e}")
            # Remove the user message we just added since we failed
            state.chat_history.pop()
            return

    # Normalize response
    answer_text = resp.get("answer", "")
    sources = resp.get("sources", [])
    # Save to session chat history
    state.chat_history.append({"role": "assistant", "text": answer_text, "sources": sources})


# ----------------------------
# Streamlit app layout
# ----------------------------

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    ensure_session_state()

    st.title(APP_TITLE)
    st.caption("Ask questions about your local documents — all offline.")

    # Sidebar: Document manager and settings
    with st.sidebar:
        st.title("📂 Documents")
        st.caption("Manage local documents in your RAG database.")
        st.divider()

        # Backend URL setting
        new_backend_url = st.text_input(
            "Backend URL",
            value=state.backend_url,
            help="URL of the FastAPI backend (e.g., http://localhost:8000)"
        )
        if new_backend_url != state.backend_url:
            state.backend_url = new_backend_url
            st.experimental_rerun()

        st.divider()

        # Health check button
        if st.button("Check Backend Health"):
            with st.spinner("Checking backend..."):
                health = check_backend_health()
            if health.get("status") == "healthy":
                st.success("Backend is healthy")
            else:
                st.error(f"Backend unhealthy: {health.get('error', 'Unknown error')}")

            st.divider()

        # Upload documents
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or Markdown files to add to the knowledge base"
        )
        if uploaded_files:
            saved_paths = []
            for f in uploaded_files:
                try:
                    saved = save_uploaded_file(f)
                    saved_paths.append(saved)
                except Exception as e:
                    st.error(f"Failed to save {f.name}: {e}")

            if saved_paths:
                with st.spinner("Ingesting documents..."):
                    result = add_documents_to_backend(saved_paths)
                    if result.get("ok", False):
                        st.success(f"✅ {result.get('message', 'Documents ingested successfully')}")
                        state.doc_count = None  # Force refresh
                    else:
                        st.warning(result.get("msg", "Ingestion completed with warnings"))

        st.divider()

        # List existing documents
        st.subheader("Indexed documents")
        docs_list = list_documents_from_backend()
        if docs_list:
            st.write(f"**{len(docs_list)} document(s) in docs/:**")
            for d in docs_list:
                name = d.get("name") or d.get("filename") or "unknown"
                st.markdown(f"- {name}")
        else:
            st.info("No documents found in docs/ folder.")

        st.divider()

        # Delete / Rebuild
        st.subheader("Delete / Rebuild")
        name_to_delete = st.text_input("Delete by filename", placeholder="example.pdf")
        if st.button("Delete document"):
            if not name_to_delete:
                st.warning("Enter filename to delete.")
            else:
                res = delete_document_from_backend(name_to_delete)
                if res.get("ok"):
                    st.success(res.get("msg"))
                    state.doc_count = None
                else:
                    st.error(res.get("msg"))

        if st.button("Rebuild index (from docs/)"):
            with st.spinner("Rebuilding index..."):
                res = rebuild_index_backend()
                if res.get("ok"):
                    st.success(res.get("msg"))
                else:
                    st.warning(res.get("msg"))

        st.divider()
        st.caption("Docs directory:")
        st.write(str(DOCS_DIR))

    # Main panel: Chat and conversation
    left, right = st.columns([3, 1])
    with left:
        st.header("Chat")

        # Conversation history display
        if state.chat_history:
            for msg in state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['text']}")
                else:
                    st.markdown("**Assistant:**")
                    st.write(msg.get("text", ""))
                    # Inline sources for existing history items (if any)
                    if msg.get("sources"):
                        with st.expander("View sources for this answer"):
                            for s in msg.get("sources", []):
                                name = s.get("id", "source")
                                snippet = s.get("content", "")[:300]
                                score = s.get("score", 0.0)
                                st.caption(f"{name} — Score: {score:.2f}")
                                st.write(snippet)
        else:
            st.info("No messages yet. Ask a question in the box below.")

        st.markdown("---")
        query_input = st.text_area("Ask your assistant", key="query_input", height=100, value=state.last_query)
        query_cols = st.columns([1, 1, 1])
        with query_cols[0]:
            if st.button("Send"):
                q = query_input.strip()
                if not q:
                    st.warning("Type a question before sending.")
                else:
                    state.last_query = q
                    query_backend_and_render(q)
        with query_cols[1]:
            if st.button("Clear chat"):
                state.chat_history = []
                st.rerun()
        with query_cols[2]:
            if st.button("Regenerate last answer"):
                # Try to re-run the last user message
                last_user_msg = None
                for m in reversed(state.chat_history):
                    if m["role"] == "user":
                        last_user_msg = m["text"]
                        break
                if not last_user_msg:
                    last_user_msg = state.last_query
                if not last_user_msg:
                    st.warning("No message to regenerate.")
                else:
                    query_backend_and_render(last_user_msg)

    with right:
        st.header("Session")
        st.write(f"Backend URL: {state.backend_url}")
        # Document count
        if state.doc_count is None:
            try:
                docs = list_documents_from_backend()
                state.doc_count = len(docs)
            except Exception:
                state.doc_count = "unknown"
        st.write("Documents in docs/:", state.doc_count)
        st.markdown("---")
        st.header("Assistant Settings")
        # Placeholder toggles for future options
        st.checkbox("Show confidence bars", value=True, key="show_confidence")
        st.checkbox("Show full explainability", value=True, key="show_explain")
        st.selectbox("Response length", ["short", "medium", "long"], index=1, key="resp_length")

        st.markdown("---")
        st.caption("TAI — Local. All models run offline. Use responsibly.")

    # Footer notes
    st.markdown("---")
    st.caption("🧠 Powered by TAI Local RAG Engine — all processing done offline via backend API.")


if __name__ == "__main__":
    main()