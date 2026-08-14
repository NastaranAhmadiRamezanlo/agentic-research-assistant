from pathlib import Path


def create_chunks(text, chunk_size=30, overlap=5):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def load_documents(documents_path):
    documents = []

    files = documents_path.glob("*.txt")

    for file in files:
        text = file.read_text(encoding="utf-8")

        documents.append(text)

    return documents


# test
documents = load_documents(Path("documents"))

for document in documents:
    chunks = create_chunks(document)

    for chunk in chunks:
        print("---- CHUNK ----")
        print(chunk)