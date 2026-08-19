"""Small reproducible evaluation set for the research assistant."""

from rag import retrieve_evidence


EVAL_QUESTIONS = [
    {
        "question": "What factors affect the efficiency of solar photovoltaic systems?",
        "expected_topics": ["solar photovoltaic", "efficiency", "photovoltaic"],
    },
    {
        "question": "What research topics are important for battery energy storage?",
        "expected_topics": ["battery", "energy storage", "photovoltaic"],
    },
]


def evaluate_retrieval(k=5):
    rows = []

    for item in EVAL_QUESTIONS:
        results = retrieve_evidence(item["question"], k=k)
        text = " ".join(result["text"].lower() for result in results)

        matched = [
            topic for topic in item["expected_topics"]
            if topic.lower() in text
        ]

        rows.append({
            "question": item["question"],
            "matched_topics": matched,
            "topic_recall": len(matched) / len(item["expected_topics"]),
            "top_score": results[0]["score"] if results else 0.0,
        })

    return rows


if __name__ == "__main__":
    for row in evaluate_retrieval():
        print(row)
