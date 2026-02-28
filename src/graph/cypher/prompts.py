"""
Few-shot Cypher prompt and examples (Strategy #3).
"""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate

FEW_SHOT_EXAMPLES = """\
Example 1 – List all movies:
  Question: What movies are in the database?
  Cypher:   MATCH (m:Movie) RETURN m.title

Example 2 – Actors in a specific movie:
  Question: Who acted in The Matrix?
  Cypher:   MATCH (a:Person)-[:ACTED_IN]->(m:Movie {{title: "The Matrix"}}) RETURN a.name

Example 3 – Director of a specific movie:
  Question: Who directed The Matrix?
  Cypher:   MATCH (d:Person)-[:DIRECTED]->(m:Movie {{title: "The Matrix"}}) RETURN d.name

Example 4 – Movies for a given actor:
  Question: What movies did Tom Hanks act in?
  Cypher:   MATCH (p:Person {{name: "Tom Hanks"}})-[:ACTED_IN]->(m:Movie) RETURN m.title

Example 5 – Multi-hop (actor → movie → director):
  Question: Who directed movies that Keanu Reeves acted in?
  Cypher:   MATCH (a:Person {{name: "Keanu Reeves"}})-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person) RETURN DISTINCT d.name, m.title"""

ENHANCED_CYPHER_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=(
        "Task: Generate a Cypher statement to query a graph database.\n"
        "Instructions:\n"
        "Use only the provided relationship types and properties in the schema.\n"
        "Do not use any other relationship types or properties that are not provided.\n"
        "\n"
        "Schema:\n{schema}\n\n"
        "Rules:\n"
        "- Output ONLY the Cypher statement — no explanations, no apologies.\n"
        "- Always include a RETURN clause.\n"
        "- If the question contains pronouns like 'those', 'these', 'them', or "
        "'they', interpret them using the most relevant entities in the schema. "
        "Never refuse to generate Cypher — always produce your best attempt.\n"
        "- Prefer explicit property lookups over full-text search.\n\n"
        "Few-shot examples:\n"
        + FEW_SHOT_EXAMPLES
        + "\n\nThe question is:\n{question}"
    ),
)
