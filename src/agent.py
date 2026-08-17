from json import tool

from tools import calculator
from tools import rag_tool

import json
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from rag import ask

load_dotenv()

client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

def choose_tool(question):

    prompt = f"""
You are a tool-selection agent.

You have two tools:

1. calculator
   Use this for mathematical calculations.
   Input should be a mathematical expression.

2. rag
   Use this for questions that can be answered using the university documents.
   Input should be the user's question.

Return ONLY valid JSON.

For calculator, use this format:
{{"tool": "calculator", "input": "12000 * 2"}}

For rag, use this format:
{{"tool": "rag", "input": "How long does the university program last?"}}

Question:
{question}

JSON:
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

    return json.loads(response.choices[0].message.content.strip())

def agent(question):

    decision = choose_tool(question)

    tool = decision["tool"]
    tool_input = decision["input"]

    if tool == "calculator":
        result = calculator(tool_input)

    elif tool == "rag":
        result = rag_tool(tool_input)

    else:
        return "I could not determine which tool to use."

    return result

if __name__ == "__main__":

    questions = [
        "How long does the university program last?",
        "Are you able to multiply 4 by 2?",
    ]

    for question in questions:

        print("\nQuestion:", question)

        decision = choose_tool(question)

        print("Tool:", decision["tool"])
        print("Input:", decision["input"])



def run_tool(decision):

    tool = decision["tool"]
    input_data = decision["input"]

    if tool == "calculator":
        return calculator(input_data)

    elif tool == "rag":
        return ask(input_data)

    else:
        return "Unknown tool"

def generate_final_answer(question, tool_result):

    prompt = f"""
You are a helpful research assistant.

Answer the user's question using the tool result below.

Do not invent information.
If the tool result does not contain enough information, say so.

User question:
{question}

Tool result:
{tool_result}

Final answer:
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




if __name__ == "__main__":

    questions = [
        "How long does the university program last?",
        "Can you multiply 12 by 2? yes? if so please do it for me.",
    ]

    for question in questions:

        decision = choose_tool(question)

        print("\nQuestion:", question)
        print("Tool:", decision["tool"])
        print("Input:", decision["input"])

        result = run_tool(decision)

        print("Tool result:", result)

        final_answer = generate_final_answer(
            question,
            result
        )

        print("Final answer:", final_answer)           