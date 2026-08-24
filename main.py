"""Document loading and chunking for the Agentic Research Assistant."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import List, Dict, Any


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    doc_type: str
    text: str


def load_documents(documents_path: Path) -> List[Dict[str, Any]]:
    """Load TXT documents and optional paper metadata from data/papers.json."""
    documents = []

    for file in sorted(documents_path.glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        documents.append(asdict(Document(
            doc_id=file.stem,
            title=file.stem.replace("_", " ").title(),
            source=str(file),
            doc_type="document",
            text=text,
        )))

    return documents


def create_chunks(
    document: Dict[str, Any],
    chunk_size: int = 120,
    overlap: int = 20,
) -> List[Dict[str, Any]]:
    """Split a document into overlapping word chunks while preserving source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = document["text"].split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size

        chunks.append({
            "chunk_id": f'{document["doc_id"]}_chunk_{chunk_id}',
            "doc_id": document["doc_id"],
            "title": document["title"],
            "source": document["source"],
            "doc_type": document["doc_type"],
            "text": " ".join(words[start:end]),
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def load_paper_metadata(path: Path = Path("data/papers.json")) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    docs = load_documents(Path("documents"))
    print(f"Loaded {len(docs)} documents.")
    for doc in docs:
        print(f'- {doc["doc_id"]}: {len(doc["text"].split())} words')
