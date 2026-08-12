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


documents_path = Path("documents")

files = documents_path.glob("*.txt")

for file in files:
    text = file.read_text()

    chunks = create_chunks(text)

    for chunk in chunks:
        print("---- CHUNK ----")
        print(chunk)