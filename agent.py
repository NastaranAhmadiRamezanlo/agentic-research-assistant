"""Agentic research workflow.

The agent plans a sequence of research actions instead of selecting only one tool.
It can retrieve semantic evidence and inspect the citation graph before generation.
"""

import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from tools import semantic_search, citation_search, graph_stats
from generator import generate_answer

load_dotenv()

client = InferenceClient(
    provider="auto",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "Qwen/Qwen3-8B"


def plan_research(question: str):
    prompt = f"""
You are a research-planning agent for a scholarly renewable-energy assistant.

Available actions:
- semantic_search: retrieve relevant document chunks.
- citation_search: inspect papers connected to a paper ID found in retrieval.
- graph_stats: inspect knowledge-graph statistics.

Create a short multi-step plan. Start with semantic_search.
If retrieval finds useful paper IDs, the program may inspect their citation
relationships. Return ONLY valid JSON in this format:

{{"steps": [{{"tool": "semantic_search", "input": "{question}"}}]}}

You may include at most 3 steps.
Do not invent paper IDs.

Question:
{question}
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()

    # Qwen can sometimes surround JSON with markdown fences.
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def run_agent(question: str):
    """Execute retrieval -> graph exploration -> grounded generation."""
    evidence = semantic_search(question, k=5)

    # Explore citation relationships for retrieved documents that exist in the graph.
    graph_evidence = []
    for item in evidence[:3]:
        try:
            related = citation_search(item["doc_id"])
            if related:
                graph_evidence.append({
                    "paper_id": item["doc_id"],
                    "related": related,
                })
        except Exception:
            pass

    # Keep the final evidence grounded in actual retrieved text.
    answer_evidence = [
        {
            "source_id": item["doc_id"],
            "title": item["title"],
            "text": item["text"],
            "score": item["score"],
        }
        for item in evidence
    ]

    answer = generate_answer(question, answer_evidence)

    return {
        "question": question,
        "retrieved_evidence": evidence,
        "citation_graph": graph_evidence,
        "answer": answer,
    }


if __name__ == "__main__":
    question = "What factors affect the efficiency of solar photovoltaic systems?"
    result = run_agent(question)

    print("\nQUESTION:\n", question)
    print("\nANSWER:\n", result["answer"])
    print("\nCITATION GRAPH EXPLORATION:")
    for item in result["citation_graph"]:
        print(item)
