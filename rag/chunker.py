"""STEP 2 (ingest): CHUNK — split long text into small overlapping pieces.

Why? Embedding models have input limits, and retrieving a focused 800-char
passage is far more precise than retrieving a whole 50-page document.
Overlap keeps a sentence that straddles a boundary intact in at least one chunk.
"""
import re


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    overlap = min(overlap, size // 2)

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer to cut at a natural boundary in the back half of the window.
            window = text[start:end]
            for sep in ("\n\n", ". ", "\n", " "):
                idx = window.rfind(sep)
                if idx > size * 0.5:
                    end = start + idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Step back by `overlap`, then snap forward to a word boundary.
        start = end - overlap
        space = text.find(" ", start, end)
        start = space + 1 if space != -1 else start
    return chunks
