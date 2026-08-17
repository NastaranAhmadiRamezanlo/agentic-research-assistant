# Prompt + LLM + Answer
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()


client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)


def generate_answer(question, context):

    prompt = f"""
You are a research assistant.

Answer the question using ONLY the information provided in the context.

If the answer cannot be found in the context, say:
"I don't have enough information to answer this question based on the provided documents."

Do not use your own knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content