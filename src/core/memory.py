#src/core/memory.py
import json, os
from typing import List, Dict

class ChatMemory:
    """
    Lightweight conversational memory manager.
    Stores user queries + bot responses in a local JSON file.
    """

    def __init__(self, db_path: str = "./chat_memory.json"):
        self.db_path = db_path
        self.history: List[Dict[str, str]] = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def add(self, query: str, response: str):
        """Add a new user–bot interaction."""
        self.history.append({"query": query, "response": response})
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def get_context(self, last_n: int = 3) -> str:
        """Return the last N interactions as a formatted string for LLM context."""
        recent = self.history[-last_n:]
        context = ""
        for entry in recent:
            # backward-compatible key access
            user_msg = entry.get("query") or entry.get("user") or ""
            assistant_msg = entry.get("response") or entry.get("assistant") or ""
            context += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
        return context.strip()


    def clear(self):
        """Clear memory if needed."""
        self.history = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
