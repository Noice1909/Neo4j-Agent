# Neo4j Multi-Agent System

**Enterprise-grade Neo4j query agent** with FastAPI, LangGraph, and FastMCP. Generates accurate Cypher queries through semantic schema understanding and multi-layer entity resolution.

## 🌟 Key Features

### Schema-Aware Intelligence
- **Schema Semantic Layer**: One-time LLM analysis at startup generates natural language metadata for every property and relationship
- **Property & Relationship Synonym Maps**: 3-layer property resolution (pattern-based → semantic layer → concept metadata)
- **Schema Reasoning**: Per-query ambiguity detection and resolution
- **Entity Hints**: Automatic entity-label association discovery from database

### Entity Resolution Pipeline (4 Layers)
- **Layer 0.5**: Concept full-text index lookup for semantic label mapping
- **Layer 1**: Label synonym resolution (3 layers: overrides → pattern-based → concept metadata)
- **Layer 2**: Entity name fuzzy search (APOC multi-signal similarity + phonetic matching)
- **Layer 2.5**: Entity label lookup (discovers which label entities belong to)
- **Layer 3**: LLM fallback for complex corrections

### Multi-Agent Architecture (10 Specialized Agents)
1. **Supervisor** — Routes queries and orchestrates workflow
2. **Coreference Resolution** — Resolves pronouns and references
3. **Entity Resolution** — 4-layer entity correction pipeline
4. **Topology Filter** — Schema reasoning and ambiguity resolution
5. **Cypher Generation** — Creates queries with dynamic few-shot examples
6. **Cypher Validation** — 4-stage validation (read-only, syntax, relationships, schema)
7. **Cypher Execution** — Executes with retry and error handling
8. **Result Verification** — Validates query results
9. **Retry Decision** — Analyzes failures and creates correction guidance
10. **Synthesis** — Generates natural language responses

### Production Features
- **Dual Backend Support**: Memory (SQLite checkpointer) or Redis (distributed caching)
- **LLM Cache**: Deduplicates identical queries to reduce latency
- **Query Deduplicator**: Prevents concurrent duplicate queries
- **Dynamic Few-Shot Examples**: 20 Cypher patterns with topology-aware filtering
- **Conversation Memory**: Session-based context tracking
- **Observability**: Prometheus metrics + structured logging
- **FastMCP Server**: Model Context Protocol for AI integration

## 🏗️ Architecture

### Single Unified Codebase
```
src/
├── agent/                  # Multi-agent pipeline (LangGraph)
│   ├── factory.py         # Agent initialization
│   ├── pipeline.py        # 10-agent workflow
│   ├── supervisor.py      # Query routing
│   └── specialists/       # Individual agents
├── graph/                 # Neo4j + Schema intelligence
│   ├── topology.py        # GraphTopology model
│   ├── schema_cache.py    # Unified in-memory/Redis cache
│   ├── semantic_layer.py  # Schema semantic layer generation
│   └── cypher/
│       ├── prompts.py             # Cypher generation prompts
│       ├── dynamic_examples.py    # 20 few-shot patterns
│       ├── schema_reasoning.py    # Ambiguity detection + resolution
│       ├── synonyms.py            # Property/relationship synonym maps
│       ├── coreference.py         # Pronoun resolution regex
│       └── entity_resolution/     # 4-layer entity pipeline
├── core/                  # Configuration and infrastructure
│   ├── config.py          # Settings (cache_backend, use_redis)
│   ├── tracing.py         # Structured event tracing
│   └── mcp_server.py      # FastMCP server
└── main.py                # FastAPI application entry point

scripts/
├── load_sample_movies.py          # Sample data loader
└── print_sample_data_cypher.py    # Manual data loading helper
```

### Cache Backend Architecture
Controlled by `CACHE_BACKEND` environment variable:

**Memory Mode** (`CACHE_BACKEND=memory`, default):
- In-memory schema cache + LLM cache
- SQLite checkpointer for conversation state
- No Redis dependency — single-instance deployment

**Redis Mode** (`CACHE_BACKEND=redis`):
- Redis for schema cache, LLM cache, query deduplication
- Redis checkpointer for distributed conversation state
- Supports multi-instance horizontal scaling

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd neo4j

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt
```

### 2. Configuration

Create `.env` file:

```bash
# Neo4j Configuration (Aura or local)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=your-username
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# LLM Configuration (Ollama local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b

# Cache Backend ("memory" or "redis")
CACHE_BACKEND=memory

# Redis Configuration (only if CACHE_BACKEND=redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=neo4j-agent
```

### 3. Load Sample Data

**Option A: Via Neo4j Aura Browser Console**
```bash
# Print Cypher to copy-paste
python scripts/print_sample_data_cypher.py
```

**Option B: Via Python Script** (if local Neo4j or SSL configured)
```bash
python -m scripts.load_sample_movies
```

**Sample Data Includes:**
- 5 Actors (Tom Hanks, Keanu Reeves, etc.)
- 4 Directors (Christopher Nolan, etc.)
- 8 Movies (Forrest Gump, The Matrix, Interstellar, etc.)
- Concept metadata nodes for NLP (Actor, Director, Movie)

### 4. Start Server

```bash
# Development mode (auto-reload)
uvicorn src.main:app --reload --port 8000

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server runs at: http://localhost:8000

## 📚 API Usage

### Chat Endpoint

**POST** `/chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Tom Hanks movies",
    "session_id": "user-123"
  }'
```

**Response:**
```json
{
  "response": "Tom Hanks has acted in 4 movies in our database:\n1. Forrest Gump (1994)...",
  "session_id": "user-123"
}
```

### Example Queries

```bash
# Entity queries (with entity hints)
"Tell me about Tom Hanks movies"
"Show me Keanu Reeves filmography"

# Aggregation queries
"How many movies are there?"
"Which director has directed the most movies?"

# Complex queries
"List top 5 actors by number of movies"
"Which movies were released after 2000?"
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:8000/health
```

## 🔧 Development

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_entity_resolution.py -v
```

### Code Quality

```bash
# Format code
ruff format src/

# Lint code
ruff check src/ --fix

# Type checking
mypy src/
```

### Batch Testing

Test multiple queries at once:

```bash
python scripts/batch_chat.py
```

## 📊 Monitoring

### Metrics Endpoint

**GET** `/metrics` — Prometheus metrics

```bash
curl http://localhost:8000/metrics
```

**Key Metrics:**
- `neo4j_agent_requests_total` — Total requests
- `neo4j_agent_request_duration_seconds` — Request latency
- `neo4j_agent_cache_hits_total` — LLM cache hits
- `neo4j_agent_cypher_validation_failures_total` — Validation failures

### Structured Logging

Logs include structured fields for tracing:
- `event` — Event type (e.g., `CYPHER_GENERATED`)
- `status` — `ok`, `fail`, `skip`, `info`
- `detail` — Event details
- `session_id` — User session identifier
- `turn_id` — Conversation turn identifier

## 🧠 How It Works

### Startup Sequence (src/main.py)

1. **Neo4j Init** — Connect to database
2. **Schema Cache Warm-up** — Extract topology (labels, relationships, properties)
3. **Semantic Layer Generation** — ONE LLM call analyzes entire schema
4. **Coreference Regex** — Build pattern from label + NLP terms
5. **Checkpointer** — Initialize SQLite or Redis checkpointer
6. **LLM Cache + Query Deduplicator** — Set up caching layers
7. **Agent Initialization** — Build 10-agent LangGraph pipeline

### Query Flow (Per Request)

1. **Supervisor** → Route to `graph_query` path
2. **Coreference** → Resolve pronouns ("his movies" → "Tom Hanks movies")
3. **Entity Resolution**:
   - Layer 0.5: Concept FT lookup ("movies" → "Movie")
   - Layer 1: Label synonym resolution
   - Layer 2: Entity name fuzzy search
   - **Layer 2.5: Entity label lookup** ("Tom Hanks" → Actor)
   - Layer 3: LLM fallback (if needed)
4. **Topology Filter** → Schema reasoning + ambiguity resolution
5. **Cypher Generation** → Build query with:
   - Dynamic few-shot examples
   - Schema context hints
   - **Entity hints** (e.g., "Tom Hanks is an Actor")
6. **Cypher Validation** → 4-stage validation
7. **Cypher Execution** → Execute with retry
8. **Result Verification** → Validate results
9. **Retry Decision** → Retry if needed with correction guidance
10. **Synthesis** → Generate natural language response

## 🎯 Key Innovations

### Entity Hints System (Layer 2.5)
**Problem**: When entity resolution finds an exact match (e.g., "Tom Hanks"), no correction is created. The Cypher generator doesn't know which label the entity belongs to.

**Solution**: After entity resolution, extract capitalized names from the original question and query Neo4j to discover their labels. Pass this as hints to Cypher generation:
```
Entity hints (names found in the database):
- "Tom Hanks" is a Actor (matched on name property)
```

**Result**: LLM generates correct query `MATCH (a:Actor {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)`

### Schema Semantic Layer
**Why**: User asks "show me thriller movies" but DB has `movie_type` property, not `genre`.

**Solution**: At startup, ONE LLM call analyzes the entire schema and generates:
- `PropertySemantics`: natural_names, description, data_type_hint for every property
- `RelationshipSemantics`: natural_phrases, description for every relationship

**Cached**: Schema hash-based caching (memory or Redis)

**Used by**:
- Property synonym maps (Layer 1.5)
- Schema reasoning (topology filter)
- Cypher prompt enhancement

### Dynamic Few-Shot Examples
20 Cypher patterns (6756 chars → reduced to 15 examples for local LLM):
- Basic MATCH
- Property filters
- Relationship traversal
- Aggregation (COUNT, AVG, MAX)
- Sorting (ORDER BY + LIMIT)
- COALESCE for optional properties
- EXISTS subqueries
- Negation with NOT
- Multi-hop paths
- Date/numeric comparisons

**Filtered by**: Available labels/relationships in filtered topology

## 🐛 Troubleshooting

### Query Returns "I couldn't find any information"

**Cause**: Cypher executed successfully but returned 0 results

**Debug**:
1. Check trace logs for generated Cypher
2. Run Cypher manually in Neo4j browser
3. Verify data exists: `MATCH (n) RETURN count(n)`
4. Check entity hints in trace: `ENTITY_HINTS` event should show entity-label associations

### "Tell me about Tom Hanks movies" fails

**Verify**:
```cypher
-- 1. Tom Hanks exists?
MATCH (a:Actor {name: 'Tom Hanks'}) RETURN a;

-- 2. Relationships exist?
MATCH (a:Actor {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)
RETURN m.title;
```

**If no relationships**: Data was loaded incorrectly (variables lost between CREATE statements). Re-run relationship creation using MATCH-based Cypher.

### SSL Certificate Error (Neo4j Aura)

**Symptom**: `ssl.SSLCertVerificationError: certificate verify failed`

**Solution**: Load data via Neo4j Aura browser console instead of Python script:
```bash
python scripts/print_sample_data_cypher.py
# Copy output and paste into Aura browser
```

## 📝 Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:14b` | Ollama model name |
| `CACHE_BACKEND` | `memory` | Cache backend: `memory` or `redis` |
| `REDIS_HOST` | `localhost` | Redis host (if `CACHE_BACKEND=redis`) |
| `REDIS_PORT` | `6379` | Redis port |
| `CHECKPOINTER_BACKEND` | `sqlite` | Checkpointer: `sqlite` or `redis` |
| `ENTITY_RESOLUTION_ENABLED` | `true` | Enable entity resolution |
| `ENTITY_FUZZY_THRESHOLD` | `0.75` | Fuzzy match threshold (0.0-1.0) |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangChain/LangGraph** — Multi-agent orchestration framework
- **Neo4j** — Graph database and Cypher query language
- **FastAPI** — Modern Python web framework
- **Ollama** — Local LLM inference
- **FastMCP** — Model Context Protocol implementation

---

**Built with ❤️ for enterprise graph intelligence**
