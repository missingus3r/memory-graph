# Memory Graph

D3.js force-directed graph visualization for Friday's memory system.

## Features

- **Graph** — Interactive force-directed graph showing memory nodes and relationships (conversations, memories, entities)
- **Logs** — Chronological view of all conversation logs with collapsible date groups
- **Architecture** — System architecture diagram with 26 nodes and 29 connections (data/control/monitoring)
- **RAG** — Semantic search dashboard with hybrid search (cosine similarity + keyword matching)

## Stack

- **Frontend**: Single-file HTML with D3.js v7, vanilla JS
- **Backend**: Flask + SQLite memory API (`api_server.py`) on port 7777
- **RAG**: Gemini Embedding 001 (3072 dim), cosine similarity, RRF hybrid ranking

## Setup

1. Start the memory API server:
```bash
cd ~/proyectos/memory-graph
source venv/bin/activate
FRIDAY_DB_PATH=~/.claude/memory.db python3 api_server.py
```

2. Access the graph at `http://localhost:7777/graph`

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /graph` | Serve Memory Graph UI |
| `GET /conversation/recent` | Get recent conversation logs |
| `GET /memory/list` | List all memories |
| `GET /entity/search?q=` | Search entities |
| `GET /search/semantic?q=` | Semantic search (RAG) |
| `GET /search/hybrid?q=` | Hybrid search (semantic + keyword) |
| `GET /kv/<key>` | Key-value storage (node positions) |

## Architecture

The graph connects to the memory API which stores:
- **Conversations**: User/assistant/system messages with timestamps
- **Memories**: Long-term storage (user, feedback, project, reference types)
- **Entities**: People, companies, tools, concepts
- **Embeddings**: 3072-dim vectors for semantic search (Gemini Embedding 001)

## Screenshots

### Graph Tab
Force-directed graph with color-coded nodes by type. Click to expand, drag to reposition.

### Architecture Tab
Fixed-position system diagram with draggable nodes. Positions persist server-side via `/kv` endpoint.

### RAG Tab
Search dashboard showing semantic similarity scores, hybrid ranking, and embedding statistics.

---

Built by [Bruno Silveira](https://github.com/missingus3r) with [Friday](https://github.com/openclaw/openclaw) AI assistant.
