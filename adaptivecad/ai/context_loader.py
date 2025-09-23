from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import os, re

DOC_CANDIDATES = [
    "README.md",
    "ADAPTIVE_PI_AXIOMS.md",
    "ADVANCED_SHAPES.md",
    "HYPERBOLIC_GEOMETRY_IMPLEMENTATION.md",
    "MODELING_TOOLS.md",
    "PLAYGROUND_GUIDE.md",
]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@dataclass
class Section:
    path: str
    title: str
    text: str

def _read(p: str) -> str:
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _split_sections(md: str) -> List[Tuple[str,str]]:
    # very light md section splitter
    parts = re.split(r"(?m)^(#{1,3})\s+(.*)$", md)
    # parts: [pre, h#, title, body, h#, title, body, ...]
    out = []
    if parts:
        pre = parts[0].strip()
        if pre:
            out.append(("Intro", pre))
        for i in range(1, len(parts), 3):
            if i + 2 >= len(parts):
                break
            hdr = parts[i+1].strip()
            body = parts[i+2].strip()
            out.append((hdr, body))
    return out

def load_repo_sections() -> List[Section]:
    sections: List[Section] = []
    for rel in DOC_CANDIDATES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        try:
            md = _read(p)
        except Exception:
            continue
        for title, body in _split_sections(md):
            # limit very long bodies
            text = body[:3000]
            sections.append(Section(path=rel, title=title, text=text))
    return sections

def _score(q: str, txt: str) -> float:
    # super simple BM25-ish: word overlap with mild weighting
    q_terms = [w for w in re.findall(r"[a-z0-9_]+", q.lower()) if len(w) > 2]
    if not q_terms:
        return 0.0
    score = 0.0
    low = txt.lower()
    for t in q_terms:
        count = low.count(t)
        if count:
            score += 1.0 + 0.3 * (count - 1)
    return score

def retrieve_context(query: str, k: int = 4, max_chars: int = 4000) -> str:
    secs = load_repo_sections()
    scored = sorted(secs, key=lambda s: _score(query, s.text + " " + s.title), reverse=True)
    buf: List[str] = []
    total = 0
    for s in scored[:k]:
        chunk = f"[{s.path} :: {s.title}]\n{s.text}\n"
        if total + len(chunk) > max_chars:
            break
        buf.append(chunk)
        total += len(chunk)
    return "\n---\n".join(buf)
