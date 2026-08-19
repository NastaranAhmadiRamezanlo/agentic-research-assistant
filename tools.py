"""Tools exposed to the research agent."""

import ast
import operator as op

from rag import retrieve_evidence
from citation_graph import get_graph


_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression):
    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree.body)


def semantic_search(query, k=5):
    return retrieve_evidence(query, k=k)


def citation_search(paper_id):
    return get_graph().related_papers(paper_id)


def graph_stats():
    return get_graph().stats()
