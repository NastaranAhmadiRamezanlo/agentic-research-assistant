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

def retrieve(question):

    question_embedding = model.encode([question])

    similarities = model.similarity(
        question_embedding,
        chunk_embeddings
    )

    best_index = similarities.argmax().item()

    return chunks[best_index]