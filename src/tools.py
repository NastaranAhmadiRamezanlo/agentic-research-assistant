from rag import ask


def calculator(expression):
    """
    Calculate a mathematical expression.
    """
    return eval(expression)


def rag_tool(question):
    """
    Answer a question using information from the document collection.
    """
    return ask(question)


if __name__ == "__main__":

    print("Calculator:")
    print(calculator("12000 * 2"))

    print("\nRAG:")
    print(rag_tool("How long does the program last?"))