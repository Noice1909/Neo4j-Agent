# Quick Start Guide

## 1. Install Dependencies

```bash
# Core dependencies (memory backend, default)
pip install -r src/requirements.txt

# OR with Redis backend support
pip install -e ".[redis]"
```

## 2. Configure Environment

Create `.env`:

```bash
# Neo4j (required)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=your-username
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# LLM (required)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

# Cache backend (optional, default: memory)
CACHE_BACKEND=memory  # or "redis" for distributed mode
```

## 3. Load Sample Data

```bash
# Print Cypher to copy-paste into Neo4j Aura browser
python scripts/print_sample_data_cypher.py

# Then paste the output into Neo4j Aura Query tab
```

## 4. Start Server

```bash
uvicorn src.main:app --reload --port 8000
```

Visit: http://localhost:8000

## 5. Test Query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Tom Hanks movies",
    "session_id": "test-123"
  }'
```

Expected response:
```json
{
  "response": "Tom Hanks has acted in 4 movies:\n1. Forrest Gump (1994)\n2. Saving Private Ryan (1998)\n3. Cast Away (2000)\n4. The Terminal (2004)",
  "session_id": "test-123"
}
```

## 6. Run Batch Tests

```bash
python scripts/batch_chat.py
```

Should see 5/5 queries succeed with entity hints working.

## Troubleshooting

### Data Loaded But Queries Return 0 Results

**Check relationships exist:**
```cypher
MATCH (a:Actor {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)
RETURN count(*);
```

If 0, relationships weren't created. Re-run relationship creation:
```cypher
// Clear relationships
MATCH ()-[r]-() DELETE r;

// Recreate with MATCH
MATCH (tom:Actor {name: 'Tom Hanks'}), (forrest:Movie {title: 'Forrest Gump'})
CREATE (tom)-[:ACTED_IN {roles: ['Forrest Gump']}]->(forrest);
// ... (repeat for all relationships)
```

### Entity Hints Not Working

Check trace for `ENTITY_HINTS` event:
```
13   ENTITY_HINTS            ✓          1 hint(s)
```

If missing, entity label lookup failed. Check logs for "Entity label lookup failed".

### SSL Certificate Error

Can't load data via Python script? Use Aura browser instead:
```bash
python scripts/print_sample_data_cypher.py
# Copy output → Paste into Aura browser console
```

---

**Need help?** See full [README.md](README.md) for detailed documentation.
