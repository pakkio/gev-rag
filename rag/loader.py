"""STEP 1 (ingest): LOAD — turn an uploaded file into plain text."""
import io

from pypdf import PdfReader

SUPPORTED = (".txt", ".md", ".pdf")


def load_text(filename: str, data: bytes) -> dict:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return {"text": "\n\n".join(pages), "pages": len(pages)}
    if name.endswith((".txt", ".md")):
        return {"text": data.decode("utf-8", errors="ignore"), "pages": 1}
    raise ValueError(f"Unsupported file type. Use one of: {', '.join(SUPPORTED)}")
