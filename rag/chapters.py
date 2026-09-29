"""CHAPTERS — map each chunk back to the chapter it belongs to.

Two steps, because neither alone is reliable across books with different layouts:
  1. candidates(): pattern matching finds lines that *look* like chapter headings
     ("CAPITOLO XII", "CANTO V", "III.", "I • IL NUMERO 24601 ...") with their exact chunk.
     Cheap and positionally exact, but full of false positives (TOC lines, list items).
  2. One LLM call gets the candidates as "id | line | following text" and names the false
     positives to drop. It is asked for rejections, not for the chapter list: asked to list the
     real chapters, a small model kept only the top level (the 17 PARTI of Guerra e pace, not its
     ~360 chapters). Asking it to find chapters in raw text instead (while summarizing) gave
     footnote numbers and fragments as chapters on long books.
"""
import re
from bisect import bisect_right

from . import gemini_llm

# The keyword must be followed by a number or an ordinal: "parte", "canto", "libro" are also
# ordinary words that start lines of prose ("parte dall'influenza del corpo").
_ORDINAL = (r"(?:(?-i:[IVXLCDM]+)\b|\d+\b|prim[oa]|second[oa]|terz[oa]|quart[oa]|quint[oa]|sest[oa]|"
            r"settim[oa]|ottav[oa]|non[oa]\b|decim[oa]|\w+esim[oa]|ultim[oa]|unic[oa])")
_KEYWORD = re.compile(
    rf"^[ \t]*((?:(?:capitolo|canto|parte|libro|tomo|capo|chapter|giornata|novella)[ \t]+|cap\.[ \t]*)"
    rf"{_ORDINAL}.{{0,70}}?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE)
# All-caps structural lines without a number, e.g. "COMINCIA LA PRIMA GIORNATA DEL DECAMERON".
_CAPS = re.compile(r"^[ \t]*((?=[^a-z\n]{3,100}$)[^\n]*\b(?:GIORNATA|NOVELLA|EPILOGO|PROLOGO|PROEMIO|"
                   r"INTRODUZIONE|CONCLUSIONE)\b[^\n]*?)[ \t]*$", re.MULTILINE)
_ROMAN = re.compile(r"^[ \t]*([IVXLCDM]{1,8}(?:\.|[ \t]*[•\-–—:.][ \t]*\S.{0,70}?)?)[ \t]*$", re.MULTILINE)
_TOC_DOTS = re.compile(r"\.{4,}")

FILTER_SYSTEM = (
    "Below are candidate lines from the book '{book}', each as '<id> | <line> | <text that follows>'. "
    "They were found by pattern matching. Most are real starts of a part, chapter, canto, section or "
    "poem — including bare Roman numerals followed by narrative text — but some are false positives: "
    "table-of-contents or index entries, editorial notes or commentary, footnotes, list items inside "
    "the text, the publisher's front matter. Reply with the ids of the FALSE positives only, "
    "comma-separated, or 'none'. Output nothing else."
)
_PARENT = re.compile(r"^(?:parte|libro|tomo|giornata|cantica)\b", re.IGNORECASE)
_BARE_NUMERAL = re.compile(r"^[IVXLCDM]{1,8}\.?$")


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", s.lower()).split())


def candidates(chunk_texts: list[str]) -> list[dict]:
    out = []
    for i, text in enumerate(chunk_texts):
        found = [m for rx in (_KEYWORD, _ROMAN, _CAPS) for m in rx.finditer(text)
                 if not _TOC_DOTS.search(m.group(1))]
        found = list({m.start(): m for m in found}.values())  # one line can match two patterns
        if len(found) >= 4:  # a table of contents or index, not the body
            continue
        for m in sorted(found, key=lambda m: m.start()):
            line = m.group(1).strip()
            # The chunk overlap repeats a heading in the next chunk; keep the first sighting.
            if any(c["chunk"] >= i - 1 and _norm(c["line"]) == _norm(line) for c in out[-3:]):
                continue
            after = " ".join(text[m.end():m.end() + 90].split())
            out.append({"id": len(out), "chunk": i, "line": line, "after": after})
    return out


def extract(book: str, chunk_texts: list[str]) -> tuple[list[dict], float]:
    """Chapters as [{"title", "start_chunk"}] in book order, plus the LLM cost."""
    cands = candidates(chunk_texts)
    if not cands:
        return [], 0.0
    listing = "\n".join(f"{c['id']} | {c['line']} | {c['after']}" for c in cands)
    r = gemini_llm.chat(FILTER_SYSTEM.format(book=book), listing, max_tokens=4000)
    if r["error"]:
        return [], 0.0
    rejected = {int(n) for n in re.findall(r"\d+", r["answer"])}
    chapters, parent = [], None
    for c in cands:
        if c["id"] in rejected:
            continue
        title = c["line"]
        if _PARENT.match(title):
            parent = title
        elif parent and _BARE_NUMERAL.match(title):
            title = f"{parent} · {title}"  # "III" alone says nothing once the chunk is out of context
        chapters.append({"title": title, "start_chunk": c["chunk"]})
    return chapters, r["usage"]["cost_usd"]


def chapter_of(chapters: list[dict], chunk_index: int) -> str | None:
    starts = [c["start_chunk"] for c in chapters]
    i = bisect_right(starts, chunk_index) - 1
    return chapters[i]["title"] if i >= 0 else None
