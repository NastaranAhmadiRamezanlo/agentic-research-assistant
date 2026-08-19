# Agentic Research Assistant

An experimental research assistant that combines semantic retrieval, a scholarly citation graph, tool-using agents, and grounded LLM generation.

## Research domain

The current demonstration domain is **renewable energy research**. The domain is intentionally independent of any specific job posting and is used to demonstrate a general research-assistant architecture.

## Architecture

```text
Research Documents
        |
        v
Document Processing
        |
   +----+-----+
   |          |
Vector Store  Citation Graph
   |          |
   +----+-----+
        |
        v
Research Agent
        |
        v
Multi-step Retrieval
        |
        v
LLM
        |
        v
Grounded Answer + Sources
        |
        v
Evaluation
```

## Current capabilities

- Document loading and chunking
- Sentence-transformer embeddings
- Semantic top-k retrieval
- Source-aware retrieval results
- Citation graph construction with NetworkX
- Agentic tool selection
- Multi-step research workflow
- Grounded answer generation
- Basic retrieval evaluation

## Example questions

- What factors affect solar photovoltaic efficiency?
- What are important research topics in battery storage?
- How can wind forecasting support renewable energy integration?

## Important note

The included papers are demonstration records for software development and testing. They are not presented as real scholarly sources. The next iteration should ingest real papers and metadata from a legitimate scholarly data source.
