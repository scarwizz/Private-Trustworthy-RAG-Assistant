# src/data/chunker.py
import re
from typing import List, Dict

_SENTENCE_END_RE = re.compile(r'(?<=[\.\?\!\n])\s+')

def split_sentences(text: str) -> List[str]:
    # small robust splitter; keeps newlines as possible sentence ends
    sents = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    return sents

def chunk_text(
    text: str,
    max_chunk_chars: int = 1000,
    overlap_chars: int = 200,
    min_chunk_chars: int = 200,
) -> List[Dict]:
    if max_chunk_chars <= overlap_chars:
        raise ValueError("max_chunk_chars must be greater than overlap_chars")

    sentences = split_sentences(text)
    if not sentences:
        return [{"text": text, "start_char": 0, "end_char": len(text)}]

    chunks = []
    joined = ""
    sent_bounds = []
    for s in sentences:
        start = len(joined)
        joined += s + " "
        end = len(joined)
        sent_bounds.append((start, end))

    n = len(sentences)
    i = 0
    while i < n:
        cur = []
        cur_len = 0
        j = i
        start_char = sent_bounds[i][0]
        while j < n:
            s = sentences[j]
            s_len = len(s) + 1
            if cur_len + s_len <= max_chunk_chars:
                cur.append(s)
                cur_len += s_len
                j += 1
            else:
                if not cur:
                    # force at least one sentence
                    cur.append(s)
                    j += 1
                break
        chunk_text = " ".join(cur).strip()
        end_char = sent_bounds[j - 1][1] if (j - 1) < len(sent_bounds) else len(joined)
        if len(chunk_text) < min_chunk_chars and j < n:
            # try to extend by one sentence if available
            chunk_text = chunk_text + " " + sentences[j]
            end_char = sent_bounds[j][1]
            j += 1

        chunks.append({"text": chunk_text, "start_char": int(start_char), "end_char": int(end_char)})

        next_start = max(0, end_char - overlap_chars)
        # find next sentence index with start >= next_start
        next_i = j
        for k in range(i, j):
            if sent_bounds[k][0] >= next_start:
                next_i = k
                break
        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks
