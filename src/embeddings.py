from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "The tuition fee is 12,000 euros per year.",
    "How much does the university cost?",
    "The weather in Tokyo is rainy."
]

embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)

print(similarities)