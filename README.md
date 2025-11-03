# TAI Assistant - Private Trustworthy RAG Assistant

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system that runs entirely offline on your machine. No cloud services, no API calls, no data leaving your device.

## 🎯 Project Overview

**TAI (Trustworthy AI) Assistant** enables you to:
- Query your documents using natural language
- Get AI-generated answers with source citations
- Maintain complete privacy with offline operation
- Understand model confidence and explainability

## 🚀 Quick Start Guide

### Prerequisites

- **Python 3.8+** (Python 3.13+ recommended - this project uses 3.13.3)
- **8GB+ RAM** (16GB recommended for better performance)
- **Windows/Linux/MacOS**
- **Optional: CUDA-compatible GPU** for faster inference

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Private-Trustworthy-RAG-Assistant
```

### Step 2: Install Dependencies

This project uses **uv** for fast, modern Python package management. Dependencies are managed via `pyproject.toml` and `uv.lock`.

**Option A: Using UV (Recommended - Already Set Up)**
```bash
# UV is already configured in this project
# Dependencies are automatically synced via uv.lock

# To add new packages:
uv add package-name

# To sync dependencies:
uv sync
```

**Option B: Using Traditional pip + venv**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1
# Or use: venv\Scripts\activate.bat

# Activate (Linux/MacOS)
source venv/bin/activate

# Install dependencies
pip install streamlit torch transformers langchain langchain-community chromadb pandas plotly sentence-transformers
```

### Step 3: Verify Installation

```bash
# Check Python version
python --version
# Should show: Python 3.13.3 (or your version)

# Check Streamlit
streamlit --version

# List installed packages
pip list
```

### Step 4: Run the Application

```bash
# Run the Streamlit UI
streamlit run src/ui/app.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
Private-Trustworthy-RAG-Assistant/
│
├── src/
│   ├── core/                      # Core RAG pipeline
│   │   ├── __init__.py
│   │   └── pipeline.py           # Main RAG orchestration
│   │
│   ├── data/                      # Data directories
│   │   ├── raw/                  # Original documents
│   │   ├── processed/            # Processed documents
│   │   └── embeddings/           # Vector embeddings
│   │
│   ├── db/                        # Vector database storage
│   │   └── chromadb/             # ChromaDB files
│   │
│   ├── models/                    # Model storage
│   │   ├── embeddings/           # Embedding models
│   │   └── llm/                  # Language models
│   │
│   ├── retrieval/                 # Document retrieval
│   │   ├── __init__.py
│   │   └── retriever.py
│   │
│   ├── generation/                # Answer generation
│   │   ├── __init__.py
│   │   └── generator.py
│   │
│   ├── ui/                        # User interface
│   │   ├── app.py                # Main Streamlit app
│   │   └── components/           # UI components
│   │       ├── query_input.py
│   │       ├── answer_display.py
│   │       ├── sources_panel.py
│   │       └── sidebar.py
│   │
│   ├── utils/                     # Utilities
│   │   ├── config.py             # Configuration
│   │   └── logger.py             # Logging setup
│   │
│   └── tests/                     # Test files
│
├── .gitignore                     # Git ignore patterns
├── .python-version               # Python version specification
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependency versions
├── main.py                       # Main entry point
├── README.md                     # This file
└── setup_project_structure.py    # Project structure setup script
```

## 🔧 Configuration

Edit `src/utils/config.py` to customize:

```python
CONFIG = {
    "models": {
        "embedding_model": "gemma",      # Embedding model
        "llm_model": "flan-t5-base"      # Generation model
    },
    "system": {
        "use_gpu": True,                  # Enable GPU if available
        "max_context_tokens": 1024        # Max context length
    }
}
```

## 🎯 Usage

### Basic Query Flow

1. **Start the application:**
   ```bash
   streamlit run src/ui/app.py
   ```

2. **Enter your question** in the search box

3. **Review the answer** with confidence score

4. **Check sources** in the right panel

5. **Adjust settings** in the sidebar as needed

### Example Queries

- "What are the main conclusions of the research?"
- "Summarize the methodology used in the study"
- "What limitations are mentioned?"
- "Compare findings across different sections"

### UI Features

- **Query Input:** Natural language question interface with examples
- **Answer Display:** Generated responses with confidence scores
- **Source Citations:** Ranked relevant documents with relevance percentages
- **Settings Panel:** Adjustable retrieval and generation parameters
- **Query History:** Recent queries for quick re-submission
- **Offline Indicator:** Visual confirmation of local-only operation

## 🏗️ Development Phases

- ✅ **Phase 1:** Project Setup & Structure
- ✅ **Phase 2:** Data & Document Handling
- 🔄 **Phase 3:** Retrieval + Generation Pipeline (In Progress)
- 📋 **Phase 4:** Explainability & Confidence Scoring (Planned)
- ✅ **Phase 5:** Local UI (Streamlit) - **Current Phase Complete**
- 📋 **Phase 6:** Testing & Optimization (Planned)
- 📋 **Phase 7:** Documentation & Deployment (Planned)

## 🔒 Privacy & Security

- **100% Offline:** No internet connection required after setup
- **Local Storage:** All data stays on your machine
- **No Telemetry:** No usage data collected or transmitted
- **Open Source:** Fully auditable codebase
- **No External APIs:** All models run locally

## 🐛 Troubleshooting

### Streamlit Not Found

**Problem:** `streamlit: command not found`
```bash
# Verify Streamlit is installed
pip list | grep streamlit

# If not found, install it
pip install streamlit

# Or with UV
uv add streamlit
```

### Module Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`
```bash
# Make sure you're in the project root
cd Private-Trustworthy-RAG-Assistant

# Run from root directory
streamlit run src/ui/app.py
```

### Port Already in Use

**Problem:** Port 8501 already in use
```bash
# Use a different port
streamlit run src/ui/app.py --server.port 8502
```

### GPU Not Detected

**Problem:** CUDA not available
```bash
# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# If False and you have NVIDIA GPU, reinstall PyTorch with CUDA:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Development Workflow

### Working with Branches

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature-name

# Merge when ready
git checkout main
git merge feature/your-feature-name
```

### Adding New Dependencies

**With UV (Recommended):**
```bash
uv add package-name
```

**With pip:**
```bash
pip install package-name
# Then manually update pyproject.toml
```

### Adding New Components

1. Create component file in appropriate directory
2. Update `__init__.py` if needed
3. Add imports to relevant modules
4. Update this README with new features
5. Add tests in `src/tests/`

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📦 Key Dependencies

Current project uses:
- **Python:** 3.13.3
- **Streamlit:** 1.50.0 - Web UI framework
- **PyTorch:** 2.7.1 - Deep learning framework
- **Transformers:** 4.54.0 - Hugging Face models
- **LangChain:** For RAG orchestration
- **ChromaDB:** Vector database
- **Pandas:** Data manipulation

See `pip list` output for complete dependency list.

## 🎓 Models Used

- **Embeddings:** Gemma 300D (sentence-transformers)
- **Generation:** Flan-T5-base (Google)
- **Vector DB:** ChromaDB

All models downloaded automatically on first run.

## 📊 System Requirements

**Minimum:**
- 8GB RAM
- 10GB free disk space
- CPU: 4 cores

**Recommended:**
- 16GB RAM
- 20GB free disk space
- GPU: NVIDIA with 4GB+ VRAM
- CPU: 8 cores

**Current Development System:**
- Python 3.13.3
- Windows with PowerShell
- UV package manager

## 🚀 Current Status (Phase 5)

✅ **Completed:**
- Project structure setup
- UI components (query input, answer display, sources panel, sidebar)
- Mock data for testing UI without pipeline
- Streamlit app with modular design
- Configuration management
- Logging system

🔄 **In Progress:**
- RAG pipeline integration (Phase 3)
- Document processing and embedding

📋 **Next Steps:**
- Connect UI to actual RAG pipeline
- Implement document upload and processing
- Add confidence scoring
- Enable GPU acceleration

## 💡 Tips for New Contributors

1. **UI is Ready:** The Streamlit interface works with mock data - you can test it immediately
2. **Pipeline Integration:** When Phase 3 is complete, update `process_query()` in `app.py`
3. **Mock Mode:** Current UI uses `get_mock_response()` for testing
4. **Component-Based:** Each UI element is modular and can be updated independently
5. **Settings Work:** All sidebar settings are stored in session state

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review existing GitHub issues
3. Create new issue with details

## 🙏 Acknowledgments

- Built with Streamlit, LangChain, and Hugging Face Transformers
- Inspired by privacy-first AI principles
- Designed for researchers, students, and privacy-conscious users

---

**Version:** 0.1.0
**Last Updated:** November 2025  
**Status:** Active Development  
**Package Manager:** UV
