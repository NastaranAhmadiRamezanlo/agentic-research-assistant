from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


chunks = [
    "University A offers a Master's degree in Artificial Intelligence.",
    "The program lasts two years.",
    "The tuition fee is 12,000 euros per year.",
    "The program is taught in English."
]


# Create embeddings for our chunks
chunk_embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding size:", len(chunk_embeddings[0]))

question = "How much does the university cost?"

question_embedding = model.encode([question])

similarities = model.similarity(
    question_embedding,
    chunk_embeddings
)

print("\nQuestion:")
print(question)

print("\nSimilarities:")
print(similarities)

best_index = similarities.argmax().item()

print("\nMost relevant chunk:")
print(chunks[best_index])