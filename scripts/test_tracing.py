"""Quick demo of the session tracer rendering."""
import sys, time
sys.path.insert(0, ".")

from src.core.tracing import SessionTracer, set_tracer, trace_event

tracer = SessionTracer(session_id="test-session-abc123-def456")
set_tracer(tracer)

trace_event("USER_INPUT", "info", "Who directed The Matrix?")
time.sleep(0.01)
trace_event("DEDUP_CHECK", "info", "CACHE MISS -> invoking agent")
time.sleep(0.005)
trace_event("AGENT_LLM_CALL", "info", "4 messages")
time.sleep(0.02)
trace_event("AGENT_DECISION", "info", "Tool call(s): query_graph_tool")
time.sleep(0.01)
trace_event("GRAPH_QUERY_START", "info", "Who directed The Matrix?")
time.sleep(0.005)
trace_event("COREFERENCE", "skip", "No coreferences detected")
time.sleep(0.01)
trace_event("ENTITY_RES_L1", "skip", "No label corrections")
time.sleep(0.01)
trace_event("ENTITY_RES_L2", "ok", "1 fix(es): Matrics->Matrix")
time.sleep(0.01)
trace_event("ENTITY_RES_L3", "skip", "Skipped (L1/L2 already corrected)")
time.sleep(0.02)
trace_event("CYPHER_GENERATED", "info", "MATCH (m:Movie)<-[:DIRECTED]-(d:Person) RETURN d.name")
time.sleep(0.005)
trace_event("CYPHER_VALIDATED", "ok", "Read-only check passed, Syntax OK")
time.sleep(0.03)
trace_event("CYPHER_EXECUTED", "ok", "Attempt 0 succeeded: The Matrix was directed by...")
time.sleep(0.01)
trace_event("RESPONSE", "ok", "The Matrix was directed by the Wachowskis.")

tracer.print_journey()
