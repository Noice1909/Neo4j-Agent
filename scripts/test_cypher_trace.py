"""Quick visual test: verify Cypher query wraps in the trace box."""

from src.core.tracing import SessionTracer

t = SessionTracer(session_id="test-cypher-display")
t.record("USER_INPUT", "ok", "Who directed The Matrix?")
t.record("GRAPH_QUERY_START", "info", "Who directed The Matrix?")
t.record("COREFERENCE", "skip", "No coreference needed")
t.record(
    "CYPHER_GENERATED",
    "info",
    (
        "MATCH (m:Movie {title: 'The Matrix'})<-[:DIRECTED]-(d:Person) "
        "RETURN d.name AS director, m.title AS movie, m.released AS year "
        "ORDER BY d.name"
    ),
)
t.record("CYPHER_VALIDATED", "ok", "Read-only + Syntax OK")
t.record("CYPHER_EXECUTED", "ok", "Attempt 0 succeeded: The Matrix was directed by the Wachowskis")
t.record("RESPONSE", "ok", "The Matrix was directed by the Wachowskis")
t.print_journey()
