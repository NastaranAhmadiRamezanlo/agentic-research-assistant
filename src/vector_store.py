from pathlib import Path

from main import load_documents, create_chunks
from sentence_transformers import SentenceTransformer


# Load documents
documents = load_documents(Path("documents"))


# Create chunks from the documents
chunks = []

for document in documents:
    document_chunks = create_chunks(document)
    chunks.extend(document_chunks)


print("Number of chunks:", len(chunks))


# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# Create embeddings for our chunks
chunk_embeddings = model.encode(chunks)


print("Embedding size:", len(chunk_embeddings[0]))

def retrieve(question, k=3):

    question_embedding = model.encode([question])

    similarities = model.similarity(
        question_embedding,
        chunk_embeddings
    )

    top_indices = similarities[0].argsort(descending=True)[:k]

    results = []

    for i in top_indices:
        results.append(
            (chunks[i], similarities[0][i].item())
        )

    return results


questions = [
    "How much does the university cost?",
    "What is the university's acceptance rate?"
]

for question in questions:

    print("QUESTION:", question)

    results = retrieve(question, k=3, threshold=0.5)

    if not results:
        print("No relevant information found.")
    else:
        for rank, (chunk, score) in enumerate(results, start=1):
            print(f"Rank: {rank}")
            print(f"Score: {score:.4f}")
            print(f"Chunk: {chunk}")
            print("-" * 30)

    print("=" * 50)