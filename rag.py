"""RAG orchestration layer."""

from vector_store import retrieve
from generator import generate_answer


def retrieve_evidence(question: str, k: int = 5):
    return retrieve(question, k=k)


def ask(question: str, k: int = 5):
    results = retrieve_evidence(question, k=k)

    evidence = [
        {
            "source_id": item["doc_id"],
            "title": item["title"],
            "text": item["text"],
            "score": item["score"],
        }
        for item in results
    ]

    return generate_answer(question, evidence)


if __name__ == "__main__":
    print(ask("What factors affect the efficiency of solar photovoltaic systems?"))
