"""Scholarly citation graph.

The graph is intentionally lightweight: NetworkX is enough for a GitHub
prototype and makes the research logic easy to understand. It can later be
replaced by Neo4j without changing the agent interface.
"""

from pathlib import Path
import json
import networkx as nx


class CitationGraph:
    def __init__(self, metadata_path="data/papers.json"):
        self.graph = nx.DiGraph()
        path = Path(metadata_path)

        if path.exists():
            papers = json.loads(path.read_text(encoding="utf-8"))
            for paper in papers:
                self.graph.add_node(
                    paper["id"],
                    title=paper.get("title", paper["id"]),
                    year=paper.get("year"),
                    authors=paper.get("authors", []),
                    doi=paper.get("doi"),
                )

            for paper in papers:
                for cited_id in paper.get("cites", []):
                    if cited_id in self.graph:
                        self.graph.add_edge(paper["id"], cited_id, relation="cites")

    def search_papers(self, query: str, limit: int = 5):
        """Simple metadata search; semantic retrieval is handled by VectorStore."""
        q = query.lower()
        matches = []

        for node_id, data in self.graph.nodes(data=True):
            haystack = " ".join([
                str(data.get("title", "")),
                " ".join(data.get("authors", [])),
            ]).lower()

            if any(term in haystack for term in q.split() if len(term) > 2):
                matches.append({
                    "id": node_id,
                    "title": data.get("title"),
                    "year": data.get("year"),
                    "doi": data.get("doi"),
                })

        return matches[:limit]

    def related_papers(self, paper_id: str, limit: int = 5):
        if paper_id not in self.graph:
            return []

        neighbors = list(self.graph.successors(paper_id)) + list(self.graph.predecessors(paper_id))
        results = []

        for node_id in dict.fromkeys(neighbors):
            data = self.graph.nodes[node_id]
            results.append({
                "id": node_id,
                "title": data.get("title"),
                "year": data.get("year"),
                "relation": "cites/cited-by",
            })

        return results[:limit]

    def stats(self):
        return {
            "papers": self.graph.number_of_nodes(),
            "citation_edges": self.graph.number_of_edges(),
        }


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = CitationGraph()
    return _graph
