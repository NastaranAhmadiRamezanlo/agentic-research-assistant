def create_chunks(text, chunk_size=30):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks




from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 2. Our three sentences
sentences = [
    "The tuition fee is 12,000 euros per year.",
    "How much does the university cost?",
    "The weather in Tokyo is rainy."
]


# 3. Convert the sentences into embeddings
embeddings = model.encode(sentences)


# 4. Calculate similarity between every pair
similarities = model.similarity(embeddings, embeddings)


# # 5. Print the sentences
# for i, sentence in enumerate(sentences):
#     print(f"\nSentence {i + 1}: {sentence}")


# # 6. Print the similarity matrix
# print("\nSimilarity matrix:")
# print(similarities)

print("Similarity between Sentence 1 and Sentence 2:")
print(similarities[0][1])

print("\nSimilarity between Sentence 1 and Sentence 3:")
print(similarities[0][2])


