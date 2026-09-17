from vector_store import retrieve
from generator import generate_answer


def ask(question):

    # Validate the question
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant information
    context = retrieve(question)

    # Step 2: Generate an answer using the context
    answer = generate_answer(question, context)

    return answer

if __name__ == "__main__":
    print(ask("how long does the program last?"))