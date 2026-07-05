# main.py
"""
Minimal CLI/demo for the local RAG pipeline.

Usage:
    python main.py

This will start a simple REPL where you can type queries. Type 'exit' or 'quit' to stop.
"""
import argparse
import logging
import textwrap

from src.core.pipeline import RAGPipeline
from src.utils.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def run_repl():
    pipeline = RAGPipeline()
    print("Local RAG demo (offline). Type 'exit' or 'quit' to end.\n")
    while True:
        try:
            q = input("\nYour query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        print("\nRetrieving and generating — please wait (local compute)...")
        res = pipeline.answer_query(q, verbose=True)
        answer = res.get("answer", "")
        sources = res.get("sources", [])
        debug = res.get("debug", {})

        print("\n--- Answer ---")
        print(textwrap.fill(answer, width=100))
        print("\n--- Sources ---")
        if sources:
            for idx, s in enumerate(sources, start=1):
                print(f"[{idx}] id={s.get('id')} file={s.get('filename')} score={s.get('score')}")
                snippet = s.get("text_snippet", "")
                print(textwrap.shorten(snippet, width=240, placeholder="..."))
                print("-" * 60)
        else:
            print("No sources found.")

        print("\n--- Debug ---")
        for k, v in debug.items():
            print(f"{k}: {v}")


def run_demo_once(sample_query: str = "What is in the sample document?"):
    """
    Small demo function that runs one query and asserts basic invariants.
    Useful for quick smoke tests.
    """
    pipeline = RAGPipeline()
    print("Running demo query:", sample_query)
    result = pipeline.answer_query(sample_query, verbose=True)
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result and isinstance(result["sources"], list)
    print("Answer:", result["answer"][:500])
    print("Sources count:", len(result["sources"]))
    if len(result["sources"]) == 0:
        print("Warning: no sources were returned from the index. Ensure Phase 2 ingestion has run and the vector DB is populated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG demo CLI")
    parser.add_argument("--demo", action="store_true", help="Run a single demo query then exit")
    parser.add_argument("--query", type=str, help="If --demo set, the demo query text to run")
    args = parser.parse_args()

    if args.demo:
        q = args.query or "Give me a short summary of the document."
        run_demo_once(q)
    else:
        run_repl()
