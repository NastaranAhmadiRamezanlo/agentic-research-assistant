"""LLM generation with grounded, citation-aware prompting."""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "Qwen/Qwen3-8B"


def _call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_answer(question, evidence):
    if not evidence:
        return "I don't have enough evidence in the research knowledge base to answer this question."

    formatted = []
    for item in evidence:
        formatted.append(
            f'[Source {item["source_id"]}] {item["title"]}\n'
            f'{item["text"]}'
        )

    context = "\n\n".join(formatted)

    prompt = f"""
You are a research assistant specializing in renewable energy research.

Answer the user's question ONLY using the evidence provided below.
Do not invent facts or citations.
Every substantive claim should be supported by one or more source IDs such as [Source paper_01].
If the evidence is insufficient, explicitly say what is missing.

Question:
{question}

Evidence:
{context}

Return:
1. A concise evidence-grounded answer.
2. A "Sources" section listing the source IDs and titles used.
"""
    return _call_llm(prompt)
