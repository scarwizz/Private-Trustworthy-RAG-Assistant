"""
Local UI application (Streamlit or Flask interface)
"""
import streamlit as st
from src.core.pipeline import RAGPipeline

def main():
    st.title("🔒 Private RAG Assistant")
    query = st.text_input("Ask a question:")
    if st.button("Submit") and query:
        st.write("Processing query locally...")
        # Placeholder response
        st.success("Answer will appear here once pipeline is connected.")

if __name__ == "__main__":
    main()
