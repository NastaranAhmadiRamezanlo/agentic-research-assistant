"""Semantic vector retrieval with source-aware results."""

from pathlib import Path
from sentence_transformers import SentenceTransformer

from main import load_documents, create_chunks


class VectorStore:
    def __init__(self, documents_path: str = "documents"):
        self.documents = load_documents(Path(documents_path))
        self.chunks = []

        for document in self.documents:
            self.chunks.extend(create_chunks(document))

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = self.model.encode(
            [chunk["text"] for chunk in self.chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ) if self.chunks else None

    def retrieve(self, question: str, k: int = 5):
        if not self.chunks:
            return []

        question_embedding = self.model.encode(
            [question],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarities = self.model.similarity(question_embedding, self.embeddings)[0]
        top_indices = similarities.argsort(descending=True)[:k]

        results = []
        for rank, index in enumerate(top_indices, start=1):
            chunk = self.chunks[int(index)]
            results.append({
                "rank": rank,
                "score": float(similarities[index]),
                **chunk,
            })

        return results


_store = None


def get_store():
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(question: str, k: int = 5):
    return get_store().retrieve(question, k=k)


if __name__ == "__main__":
    for result in retrieve("What factors affect solar photovoltaic efficiency?"):
        print(result)
