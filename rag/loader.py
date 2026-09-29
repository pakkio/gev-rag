"""STEP 1 (ingest): LOAD — turn an uploaded file into plain text."""
import io

from pypdf import PdfReader

SUPPORTED = (".txt", ".md", ".pdf")


def _undouble(page: str) -> str:
    """Some PDFs (e.g. Liber Liber's) draw each page's text twice, so pypdf returns
    "<text> <page number> <text again>". Left in, every passage becomes two chunks that
    crowd each other out of the top-k. Keep the first copy when the rest repeats it."""
    head = page[:120]
    k = page.find(head, 1) if len(head) >= 60 else -1
    if k == -1:
        return page
    first, second = " ".join(page[:k].split()), " ".join(page[k:].split())
    n = min(len(first), len(second)) - 20  # the first copy ends with the page number
    return page[:k] if n > len(first) // 2 and first[:n] == second[:n] else page


def load_text(filename: str, data: bytes) -> dict:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [_undouble(page.extract_text() or "") for page in reader.pages]
        return {"text": "\n\n".join(pages), "pages": len(pages)}
    if name.endswith((".txt", ".md")):
        return {"text": data.decode("utf-8", errors="ignore"), "pages": 1}
    raise ValueError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")
