#!/usr/bin/env python3
"""Friday Memory API — HTTP REST server for persistent memory.

Standalone HTTP API replacing the MCP server.
Runs on localhost:7777.

Endpoints:
  POST /conversation/log
  GET  /conversation/search?q=...&limit=20&channel=&role=
  GET  /conversation/recent?limit=50&channel=
  POST /conversation/summarize?days_old=7
  GET  /conversation/stats
  POST /memory
  GET  /memory/search?q=...&type=&limit=10
  GET  /memory/list?type=&limit=50
  DELETE /memory/<id>
  GET  /memory/recall?topic=...&limit=10
  POST /entity
  GET  /entity/search?q=...&type=&limit=20
  GET  /stats
  GET  /health
"""

import json
import os
import re
import sqlite3
import datetime
import struct
import math
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from flask import Flask, request, jsonify, g, send_file
import sqlite_utils

# ── Config ──
VERSION = "1.0.0"
DB_PATH = os.environ.get("FRIDAY_DB_PATH", str(Path.home() / ".friday" / "memory.db"))
PORT = int(os.environ.get("FRIDAY_MEMORY_PORT", "7777"))


def get_db():
    """Get a per-request database connection."""
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        g.db = sqlite_utils.Database(conn)
    return g.db


def init_db():
    """Initialize tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    db = sqlite_utils.Database(conn)

    if "conversations" not in db.table_names():
        db["conversations"].create({
            "id": int, "timestamp": str, "session_id": str,
            "role": str, "content": str, "channel": str, "metadata": str,
        }, pk="id", if_not_exists=True)
        db["conversations"].create_index(["timestamp"], if_not_exists=True)
        db["conversations"].create_index(["session_id"], if_not_exists=True)
        db["conversations"].create_index(["role"], if_not_exists=True)

    # Add importance column if missing (migration for existing DBs)
    try:
        conn.execute("ALTER TABLE conversations ADD COLUMN importance REAL DEFAULT 0.4")
    except Exception:
        pass  # Column already exists

    if "conversations_fts" not in db.table_names():
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
            USING fts5(content, content_rowid='id');
        """)

    if "memories" not in db.table_names():
        db["memories"].create({
            "id": int, "type": str, "name": str, "description": str,
            "content": str, "tags": str, "created_at": str, "updated_at": str,
        }, pk="id", if_not_exists=True)
        db["memories"].create_index(["type"], if_not_exists=True)
        db["memories"].create_index(["name"], if_not_exists=True)

    if "memories_fts" not in db.table_names():
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(name, description, content, content_rowid='id');
        """)

    if "entities" not in db.table_names():
        db["entities"].create({
            "id": int, "name": str, "type": str, "details": str,
            "created_at": str, "updated_at": str,
        }, pk="id", if_not_exists=True)
        db["entities"].create_index(["name"], if_not_exists=True)
        db["entities"].create_index(["type"], if_not_exists=True)

    # Proto-AGI tables
    for tbl_sql in [
        """CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, analysis TEXT, insights TEXT, actions TEXT, created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, trigger_pattern TEXT, description TEXT, steps TEXT,
            times_used INTEGER DEFAULT 0, last_used TEXT, created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule TEXT, source_count INTEGER DEFAULT 0, source_ids TEXT,
            confidence REAL DEFAULT 0.5, created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT, pattern TEXT, evidence TEXT, confidence REAL DEFAULT 0.5,
            valid_until TEXT, created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT, change_type TEXT, description TEXT, diff_preview TEXT,
            status TEXT DEFAULT 'pending', created_at TEXT, resolved_at TEXT
        )"""
    ]:
        try:
            conn.execute(tbl_sql)
        except Exception:
            pass

    conn.close()


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ── Importance classification ──
_URL_RE = re.compile(r"https?://")

def classify_importance(role, content):
    """Classify message importance (0.0-1.0) based on role and content keywords."""
    lower = content.lower()

    # Critical (1.0)
    if any(kw in lower for kw in ["nota ", "nota:", "save", "guarda", "remember", "recorda", "recordar", "importante"]):
        return 1.0
    # High (0.8)
    if any(kw in lower for kw in ["proyecto", "project", "compra", "pagina", "deploy", "push", "commit", "notion"]):
        return 0.8
    # Medium (0.6)
    if _URL_RE.search(lower) or any(kw in lower for kw in ["busca", "search", "investiga", "agrega", "crea"]):
        return 0.6
    # Role-based defaults
    if role == "system":
        return 0.1
    if role == "assistant":
        return 0.2
    # Default for user
    return 0.4


# ── Embedding helpers ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072

def _load_gemini_key():
    """Load Gemini API key from .env if not in environment."""
    global GEMINI_API_KEY
    if GEMINI_API_KEY:
        return GEMINI_API_KEY
    env_path = Path.home() / ".claude" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                GEMINI_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                return GEMINI_API_KEY
    return ""

def generate_embedding(text):
    """Generate embedding via Gemini API. Returns list of floats or None."""
    key = _load_gemini_key()
    if not key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent?key={key}"
    payload = json.dumps({"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text[:8000]}]}})
    req = Request(url, data=payload.encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data.get("embedding", {}).get("values")
    except Exception:
        return None

def pack_embedding(values):
    """Pack float list to bytes for SQLite BLOB storage."""
    return struct.pack(f"{len(values)}f", *values)

def unpack_embedding(blob):
    """Unpack bytes to float list."""
    n = len(blob) // 4
    return struct.unpack(f"{n}f", blob)

def cosine_similarity(a, b):
    """Cosine similarity between two float lists."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def embed_async(source_type, source_id, text):
    """Generate and store embedding in background thread."""
    def _work():
        emb = generate_embedding(text)
        if emb:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT, source_id INTEGER, embedding BLOB, text_preview TEXT, created_at TEXT)")
            conn.execute("INSERT INTO embeddings (source_type, source_id, embedding, text_preview, created_at) VALUES (?, ?, ?, ?, ?)",
                         [source_type, source_id, pack_embedding(emb), text[:200], now_iso()])
            conn.commit()
            conn.close()
    threading.Thread(target=_work, daemon=True).start()


def init_embeddings_table():
    """Create embeddings table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT, source_id INTEGER, embedding BLOB, text_preview TEXT, created_at TEXT)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_source ON embeddings (source_type, source_id)")
    conn.commit()
    conn.close()


# ── Flask App ──
app = Flask(__name__)


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.conn.close()


@app.route("/conversation/log", methods=["POST"])
def conversation_log():
    db = get_db()
    data = request.json or {}
    role = data.get("role", "")
    content = data.get("content", "")
    channel = data.get("channel", "telegram")
    session_id = data.get("session_id", "")
    metadata = data.get("metadata", "{}")

    if not role or not content:
        return jsonify({"error": "role and content required"}), 400

    ts = now_iso()
    importance = classify_importance(role, content)

    db["conversations"].insert({
        "timestamp": ts, "session_id": session_id, "role": role,
        "content": content, "channel": channel, "metadata": metadata,
        "importance": importance,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    db.execute(
        "INSERT INTO conversations_fts(rowid, content) VALUES (?, ?)",
        [last_id, content]
    )

    # Generate embedding async
    embed_async("conversation", last_id, content)

    return jsonify({"status": "ok", "id": last_id, "timestamp": ts, "importance": importance})


@app.route("/conversation/search")
def conversation_search():
    db = get_db()
    query = request.args.get("q", "")
    limit = int(request.args.get("limit", "20"))
    channel = request.args.get("channel", "")
    role = request.args.get("role", "")

    if not query:
        return jsonify({"error": "q parameter required"}), 400

    sql = """
        SELECT c.id, c.timestamp, c.role, c.content, c.channel, c.session_id
        FROM conversations c
        JOIN conversations_fts fts ON c.id = fts.rowid
        WHERE conversations_fts MATCH ?
    """
    params = [query]
    if channel:
        sql += " AND c.channel = ?"
        params.append(channel)
    if role:
        sql += " AND c.role = ?"
        params.append(role)
    sql += " ORDER BY c.timestamp DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    results = [{"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3],
                "channel": r[4], "session_id": r[5]} for r in rows]

    return jsonify({"count": len(results), "results": results})


@app.route("/conversation/recent")
def conversation_recent():
    db = get_db()
    limit = int(request.args.get("limit", "50"))
    channel = request.args.get("channel", "")

    sql = "SELECT id, timestamp, role, content, channel, session_id FROM conversations"
    params = []
    if channel:
        sql += " WHERE channel = ?"
        params.append(channel)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    results = [{"id": r[0], "timestamp": r[1], "role": r[2], "content": r[3],
                "channel": r[4], "session_id": r[5]} for r in rows]

    return jsonify({"count": len(results), "results": results})


@app.route("/conversation/summarize", methods=["POST"])
def conversation_summarize():
    """Generate weekly summaries for old logs without deleting originals."""
    db = get_db()
    data = request.json or {}
    days_old = int(data.get("days_old", 7))

    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_old)).isoformat()

    # Get old entries grouped by ISO week (exclude existing summaries)
    rows = db.execute(
        "SELECT id, timestamp, role, content FROM conversations WHERE timestamp < ? AND role != 'summary' ORDER BY timestamp",
        [cutoff]
    ).fetchall()

    if not rows:
        return jsonify({"status": "nothing to summarize", "summaries_created": 0})

    # Group by ISO week (YYYY-WNN)
    by_week = {}
    for row in rows:
        try:
            dt = datetime.datetime.fromisoformat(row[1])
            week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        except Exception:
            week_key = row[1][:10]
        by_week.setdefault(week_key, []).append(row)

    summaries_created = 0
    entries_covered = 0

    for week_key, entries in by_week.items():
        # Check if summary already exists for this week
        existing = db.execute(
            "SELECT id FROM conversations WHERE role = 'summary' AND metadata LIKE ?",
            [f'%"week":"{week_key}"%']
        ).fetchone()
        if existing:
            continue

        # Build summary from user messages (most important content)
        user_msgs = [e[3] for e in entries if e[2] == "user"]
        if not user_msgs:
            user_msgs = [e[3] for e in entries[:20]]
        summary_text = f"Weekly summary ({week_key}, {len(entries)} messages): " + " | ".join(user_msgs)
        summary_text = summary_text[:3000]

        # Get date range
        first_date = entries[0][1][:10]
        last_date = entries[-1][1][:10]
        entry_ids = [e[0] for e in entries]

        # Insert summary (originals are NOT deleted)
        ts = now_iso()
        db["conversations"].insert({
            "timestamp": ts, "session_id": "", "role": "summary",
            "content": summary_text, "channel": "system",
            "metadata": json.dumps({
                "week": week_key, "first_date": first_date, "last_date": last_date,
                "original_count": len(entries), "entry_ids": entry_ids[:100]
            }),
            "importance": 0.7,
        }, pk="id")
        summary_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute("INSERT INTO conversations_fts(rowid, content) VALUES (?, ?)", [summary_id, summary_text])
        embed_async("conversation", summary_id, summary_text)

        entries_covered += len(entries)
        summaries_created += 1

    return jsonify({"status": "ok", "entries_covered": entries_covered, "summaries_created": summaries_created})


@app.route("/conversation/stats")
def conversation_stats():
    """Get conversation statistics."""
    db = get_db()

    total = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

    # By role
    role_rows = db.execute("SELECT role, COUNT(*) FROM conversations GROUP BY role").fetchall()
    by_role = {r[0]: r[1] for r in role_rows}

    # By importance bracket
    low = db.execute("SELECT COUNT(*) FROM conversations WHERE COALESCE(importance, 0.4) < 0.4").fetchone()[0]
    medium = db.execute("SELECT COUNT(*) FROM conversations WHERE COALESCE(importance, 0.4) >= 0.4 AND COALESCE(importance, 0.4) < 0.7").fetchone()[0]
    high = db.execute("SELECT COUNT(*) FROM conversations WHERE COALESCE(importance, 0.4) >= 0.7").fetchone()[0]

    # Timestamps
    oldest = db.execute("SELECT MIN(timestamp) FROM conversations").fetchone()[0]
    newest = db.execute("SELECT MAX(timestamp) FROM conversations").fetchone()[0]

    # Summaries
    summaries = by_role.get("summary", 0)

    return jsonify({
        "total": total,
        "by_role": by_role,
        "by_importance": {"low": low, "medium": medium, "high": high},
        "oldest_timestamp": oldest,
        "newest_timestamp": newest,
        "total_summaries": summaries,
    })


@app.route("/memory", methods=["POST"])
def memory_store():
    db = get_db()
    data = request.json or {}
    name = data.get("name", "")
    content = data.get("content", "")
    mem_type = data.get("type", "project")
    description = data.get("description", "")
    tags = data.get("tags", "[]")

    if not name or not content:
        return jsonify({"error": "name and content required"}), 400

    ts = now_iso()
    existing = list(db["memories"].rows_where("name = ?", [name]))

    if existing:
        mem_id = existing[0]["id"]
        db["memories"].update(mem_id, {
            "content": content, "description": description,
            "type": mem_type, "tags": tags, "updated_at": ts,
        })
        db.execute("DELETE FROM memories_fts WHERE rowid = ?", [mem_id])
        db.execute(
            "INSERT INTO memories_fts(rowid, name, description, content) VALUES (?, ?, ?, ?)",
            [mem_id, name, description, content]
        )
        return jsonify({"status": "updated", "id": mem_id})
    else:
        db["memories"].insert({
            "type": mem_type, "name": name, "description": description,
            "content": content, "tags": tags, "created_at": ts, "updated_at": ts,
        }, pk="id")
        last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute(
            "INSERT INTO memories_fts(rowid, name, description, content) VALUES (?, ?, ?, ?)",
            [last_id, name, description, content]
        )
        # Generate embedding async
        embed_async("memory", last_id, f"{name} {description} {content}")

        return jsonify({"status": "created", "id": last_id})


@app.route("/memory/search")
def memory_search():
    db = get_db()
    query = request.args.get("q", "")
    mem_type = request.args.get("type", "")
    limit = int(request.args.get("limit", "10"))

    if not query:
        return jsonify({"error": "q parameter required"}), 400

    sql = """
        SELECT m.id, m.type, m.name, m.description, m.content, m.tags,
               m.created_at, m.updated_at
        FROM memories m
        JOIN memories_fts fts ON m.id = fts.rowid
        WHERE memories_fts MATCH ?
    """
    params = [query]
    if mem_type:
        sql += " AND m.type = ?"
        params.append(mem_type)
    sql += " LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    results = [{"id": r[0], "type": r[1], "name": r[2], "description": r[3],
                "content": r[4], "tags": r[5], "created_at": r[6], "updated_at": r[7]}
               for r in rows]

    return jsonify({"count": len(results), "results": results})


@app.route("/memory/list")
def memory_list():
    db = get_db()
    mem_type = request.args.get("type", "")
    limit = int(request.args.get("limit", "50"))

    sql = "SELECT id, type, name, description, tags, created_at, updated_at FROM memories"
    params = []
    if mem_type:
        sql += " WHERE type = ?"
        params.append(mem_type)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    results = [{"id": r[0], "type": r[1], "name": r[2], "description": r[3],
                "tags": r[4], "created_at": r[5], "updated_at": r[6]}
               for r in rows]

    return jsonify({"count": len(results), "results": results})


@app.route("/memory/<int:mem_id>", methods=["DELETE"])
def memory_delete(mem_id):
    db = get_db()
    existing = list(db["memories"].rows_where("id = ?", [mem_id]))
    if not existing:
        return jsonify({"error": f"Memory {mem_id} not found"}), 404

    db["memories"].delete(mem_id)
    db.execute("DELETE FROM memories_fts WHERE rowid = ?", [mem_id])
    return jsonify({"status": "deleted", "id": mem_id})


@app.route("/memory/recall")
def memory_recall():
    db = get_db()
    topic = request.args.get("topic", "")
    limit = int(request.args.get("limit", "10"))

    if not topic:
        return jsonify({"error": "topic parameter required"}), 400

    results = {"memories": [], "conversations": []}

    try:
        mem_rows = db.execute("""
            SELECT m.id, m.type, m.name, m.content, m.updated_at
            FROM memories m JOIN memories_fts fts ON m.id = fts.rowid
            WHERE memories_fts MATCH ? LIMIT ?
        """, [topic, limit]).fetchall()
        results["memories"] = [{"id": r[0], "type": r[1], "name": r[2],
                                 "content": r[3], "updated_at": r[4]} for r in mem_rows]
    except Exception:
        pass

    try:
        conv_rows = db.execute("""
            SELECT c.id, c.timestamp, c.role, c.content, c.channel
            FROM conversations c JOIN conversations_fts fts ON c.id = fts.rowid
            WHERE conversations_fts MATCH ?
            ORDER BY c.timestamp DESC LIMIT ?
        """, [topic, limit]).fetchall()
        results["conversations"] = [{"id": r[0], "timestamp": r[1], "role": r[2],
                                      "content": r[3], "channel": r[4]} for r in conv_rows]
    except Exception:
        pass

    return jsonify(results)


@app.route("/entity", methods=["POST"])
def entity_store():
    db = get_db()
    data = request.json or {}
    name = data.get("name", "")
    ent_type = data.get("type", "")
    details = data.get("details", "{}")

    if not name or not ent_type:
        return jsonify({"error": "name and type required"}), 400

    ts = now_iso()
    existing = list(db["entities"].rows_where("name = ? AND type = ?", [name, ent_type]))

    if existing:
        eid = existing[0]["id"]
        db["entities"].update(eid, {"details": details, "updated_at": ts})
        return jsonify({"status": "updated", "id": eid})
    else:
        db["entities"].insert({
            "name": name, "type": ent_type, "details": details,
            "created_at": ts, "updated_at": ts,
        }, pk="id")
        last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        return jsonify({"status": "created", "id": last_id})


@app.route("/entity/search")
def entity_search():
    db = get_db()
    query = request.args.get("q", "")
    ent_type = request.args.get("type", "")
    limit = int(request.args.get("limit", "20"))

    sql = "SELECT id, name, type, details, created_at, updated_at FROM entities WHERE 1=1"
    params = []
    if query:
        sql += " AND name LIKE ?"
        params.append(f"%{query}%")
    if ent_type:
        sql += " AND type = ?"
        params.append(ent_type)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    results = [{"id": r[0], "name": r[1], "type": r[2], "details": r[3],
                "created_at": r[4], "updated_at": r[5]} for r in rows]

    return jsonify({"count": len(results), "results": results})


@app.route("/stats")
def stats():
    db = get_db()
    conv_count = db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    mem_count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    ent_count = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    return jsonify({
        "conversations": conv_count,
        "memories": mem_count,
        "entities": ent_count,
        "db_size_mb": round(db_size / 1024 / 1024, 2),
        "db_path": DB_PATH,
    })


@app.route("/search/semantic")
def semantic_search():
    """Semantic search using embeddings."""
    q = request.args.get("q", "")
    limit = int(request.args.get("limit", "10"))
    source_type = request.args.get("type", "")

    if not q:
        return jsonify({"error": "query required"}), 400

    query_emb = generate_embedding(q)
    if not query_emb:
        return jsonify({"error": "embedding generation failed"}), 500

    conn = sqlite3.connect(DB_PATH)
    sql = "SELECT id, source_type, source_id, embedding, text_preview, created_at FROM embeddings"
    params = []
    if source_type:
        sql += " WHERE source_type = ?"
        params.append(source_type)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    results = []
    for row in rows:
        emb = unpack_embedding(row[3])
        score = cosine_similarity(query_emb, emb)
        results.append({
            "id": row[0], "source_type": row[1], "source_id": row[2],
            "text_preview": row[4], "created_at": row[5], "score": round(score, 4)
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"count": len(results[:limit]), "results": results[:limit]})


@app.route("/search/hybrid")
def hybrid_search():
    """Hybrid search: FTS5 + semantic, combined ranking."""
    q = request.args.get("q", "")
    limit = int(request.args.get("limit", "10"))

    if not q:
        return jsonify({"error": "query required"}), 400

    # FTS results
    conn = sqlite3.connect(DB_PATH)
    fts_rows = conn.execute(
        "SELECT c.id, c.content, c.role, c.channel, c.timestamp, COALESCE(c.importance, 0.4) FROM conversations c JOIN conversations_fts f ON c.id = f.rowid WHERE conversations_fts MATCH ? ORDER BY rank LIMIT ?",
        [q, limit * 2]
    ).fetchall()

    fts_results = {}
    for i, row in enumerate(fts_rows):
        fts_results[row[0]] = {"id": row[0], "content": row[1], "role": row[2], "channel": row[3], "timestamp": row[4], "importance": row[5], "fts_rank": i + 1}

    # Semantic results
    query_emb = generate_embedding(q)
    sem_results = {}
    if query_emb:
        emb_rows = conn.execute("SELECT source_id, embedding, text_preview FROM embeddings WHERE source_type = 'conversation'").fetchall()
        scored = []
        for row in emb_rows:
            emb = unpack_embedding(row[1])
            score = cosine_similarity(query_emb, emb)
            scored.append((row[0], score, row[2]))
        scored.sort(key=lambda x: x[1], reverse=True)
        for i, (sid, score, preview) in enumerate(scored[:limit * 2]):
            sem_results[sid] = {"source_id": sid, "score": score, "sem_rank": i + 1, "text_preview": preview}

    conn.close()

    # Combine: RRF (Reciprocal Rank Fusion) weighted by importance
    all_ids = set(fts_results.keys()) | set(sem_results.keys())
    combined = []
    for sid in all_ids:
        fts_r = fts_results.get(sid, {}).get("fts_rank", 100)
        sem_r = sem_results.get(sid, {}).get("sem_rank", 100)
        importance = fts_results.get(sid, {}).get("importance", 0.4)
        rrf_score = (1 / (60 + fts_r) + 1 / (60 + sem_r)) * (0.5 + importance)
        entry = fts_results.get(sid, {})
        if not entry and sid in sem_results:
            entry = {"id": sid, "content": sem_results[sid].get("text_preview", ""), "role": "", "channel": "", "timestamp": "", "importance": 0.4}
        entry["rrf_score"] = round(rrf_score, 6)
        entry["semantic_score"] = round(sem_results.get(sid, {}).get("score", 0), 4)
        combined.append(entry)

    combined.sort(key=lambda x: x["rrf_score"], reverse=True)
    results = combined[:limit]

    # Enrich results with associated weekly summaries
    conn2 = sqlite3.connect(DB_PATH)
    summaries_cache = {}
    for entry in results:
        ts = entry.get("timestamp", "")
        if not ts or len(ts) < 10:
            continue
        try:
            dt = datetime.datetime.fromisoformat(ts)
            week_key = f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
        except Exception:
            continue
        if week_key not in summaries_cache:
            row = conn2.execute(
                "SELECT content FROM conversations WHERE role = 'summary' AND metadata LIKE ? LIMIT 1",
                [f'%"week":"{week_key}"%']
            ).fetchone()
            summaries_cache[week_key] = row[0] if row else None
        if summaries_cache[week_key]:
            entry["weekly_summary"] = summaries_cache[week_key]
            entry["week"] = week_key
    conn2.close()

    return jsonify({"count": len(results), "results": results})


@app.route("/embeddings/stats")
def embeddings_stats():
    """Get embedding index stats."""
    conn = sqlite3.connect(DB_PATH)
    try:
        total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        by_type = conn.execute("SELECT source_type, COUNT(*) FROM embeddings GROUP BY source_type").fetchall()
        conn.close()
        return jsonify({"total": total, "by_type": {r[0]: r[1] for r in by_type}})
    except Exception:
        conn.close()
        return jsonify({"total": 0, "by_type": {}})


@app.route("/embeddings/reindex", methods=["POST"])
def embeddings_reindex():
    """Reindex all conversations and memories (background)."""
    def _reindex():
        conn = sqlite3.connect(DB_PATH)
        conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT, source_id INTEGER, embedding BLOB, text_preview TEXT, created_at TEXT)")
        # Get existing indexed IDs
        existing_conv = set(r[0] for r in conn.execute("SELECT source_id FROM embeddings WHERE source_type='conversation'").fetchall())
        existing_mem = set(r[0] for r in conn.execute("SELECT source_id FROM embeddings WHERE source_type='memory'").fetchall())
        # Index missing conversations
        rows = conn.execute("SELECT id, content FROM conversations WHERE id NOT IN ({})".format(",".join(str(i) for i in existing_conv) or "0")).fetchall()
        for row in rows:
            emb = generate_embedding(row[1])
            if emb:
                conn.execute("INSERT INTO embeddings (source_type, source_id, embedding, text_preview, created_at) VALUES (?, ?, ?, ?, ?)",
                             ["conversation", row[0], pack_embedding(emb), row[1][:200], now_iso()])
                conn.commit()
        # Index missing memories
        rows = conn.execute("SELECT id, name, description, content FROM memories WHERE id NOT IN ({})".format(",".join(str(i) for i in existing_mem) or "0")).fetchall()
        for row in rows:
            text = f"{row[1]} {row[2]} {row[3]}"
            emb = generate_embedding(text)
            if emb:
                conn.execute("INSERT INTO embeddings (source_type, source_id, embedding, text_preview, created_at) VALUES (?, ?, ?, ?, ?)",
                             ["memory", row[0], pack_embedding(emb), text[:200], now_iso()])
                conn.commit()
        conn.close()

    threading.Thread(target=_reindex, daemon=True).start()
    return jsonify({"status": "reindexing started"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": VERSION})


@app.route("/version")
def version():
    return jsonify({"version": VERSION})


@app.route("/graph")
def graph():
    graph_path = Path.home() / "proyectos" / "memory-graph" / "index.html"
    if graph_path.exists():
        return send_file(str(graph_path))
    return "Graph not found", 404


@app.route("/kv/<key>", methods=["GET", "PUT"])
def key_value(key):
    """Simple key-value store for UI state (arch positions, etc.)"""
    db = get_db()
    conn = db.conn
    conn.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")
    if request.method == "PUT":
        data = request.get_json(force=True)
        conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)", [key, json.dumps(data)])
        conn.commit()
        return jsonify({"status": "ok"})
    else:
        row = conn.execute("SELECT value FROM kv WHERE key = ?", [key]).fetchone()
        if row:
            return jsonify(json.loads(row[0]))
        return jsonify({})


# ── Proto-AGI Endpoints ──

# -- Reflections --

@app.route("/reflection", methods=["POST"])
def reflection_create():
    db = get_db()
    data = request.json or {}
    ts = now_iso()
    db["reflections"].insert({
        "date": data.get("date", ts[:10]),
        "analysis": data.get("analysis", ""),
        "insights": data.get("insights", ""),
        "actions": data.get("actions", ""),
        "created_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/reflection/recent")
def reflection_recent():
    db = get_db()
    limit = int(request.args.get("limit", "5"))
    rows = db.execute(
        "SELECT id, date, analysis, insights, actions, created_at FROM reflections ORDER BY created_at DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "date": r[1], "analysis": r[2], "insights": r[3],
                "actions": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/reflection/list")
def reflection_list():
    db = get_db()
    rows = db.execute(
        "SELECT id, date, analysis, insights, actions, created_at FROM reflections ORDER BY created_at DESC"
    ).fetchall()
    results = [{"id": r[0], "date": r[1], "analysis": r[2], "insights": r[3],
                "actions": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


# -- Skills --

@app.route("/skill", methods=["POST"])
def skill_create():
    db = get_db()
    data = request.json or {}
    ts = now_iso()
    db["skills"].insert({
        "name": data.get("name", ""),
        "trigger_pattern": data.get("trigger_pattern", ""),
        "description": data.get("description", ""),
        "steps": data.get("steps", ""),
        "times_used": 0,
        "last_used": None,
        "created_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/skill/match")
def skill_match():
    db = get_db()
    task = request.args.get("task", "")
    if not task:
        return jsonify({"error": "task parameter required"}), 400
    pattern = f"%{task}%"
    rows = db.execute(
        "SELECT id, name, trigger_pattern, description, steps, times_used, last_used, created_at FROM skills WHERE trigger_pattern LIKE ? OR description LIKE ? ORDER BY times_used DESC LIMIT 3",
        [pattern, pattern]
    ).fetchall()
    results = [{"id": r[0], "name": r[1], "trigger_pattern": r[2], "description": r[3],
                "steps": r[4], "times_used": r[5], "last_used": r[6], "created_at": r[7]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/skill/list")
def skill_list():
    db = get_db()
    rows = db.execute(
        "SELECT id, name, trigger_pattern, description, steps, times_used, last_used, created_at FROM skills ORDER BY times_used DESC"
    ).fetchall()
    results = [{"id": r[0], "name": r[1], "trigger_pattern": r[2], "description": r[3],
                "steps": r[4], "times_used": r[5], "last_used": r[6], "created_at": r[7]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/skill/<int:skill_id>/use", methods=["PUT"])
def skill_use(skill_id):
    db = get_db()
    ts = now_iso()
    db.execute("UPDATE skills SET times_used = times_used + 1, last_used = ? WHERE id = ?", [ts, skill_id])
    return jsonify({"status": "ok", "id": skill_id})


# -- Preferences --

@app.route("/preference", methods=["POST"])
def preference_create():
    db = get_db()
    data = request.json or {}
    ts = now_iso()
    db["preferences"].insert({
        "rule": data.get("rule", ""),
        "source_count": data.get("source_count", 0),
        "source_ids": data.get("source_ids", ""),
        "confidence": data.get("confidence", 0.5),
        "created_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/preference/list")
def preference_list():
    db = get_db()
    rows = db.execute(
        "SELECT id, rule, source_count, source_ids, confidence, created_at FROM preferences ORDER BY confidence DESC"
    ).fetchall()
    results = [{"id": r[0], "rule": r[1], "source_count": r[2], "source_ids": r[3],
                "confidence": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/preference/active")
def preference_active():
    db = get_db()
    rows = db.execute(
        "SELECT id, rule, source_count, source_ids, confidence, created_at FROM preferences WHERE confidence >= 0.7 ORDER BY confidence DESC"
    ).fetchall()
    results = [{"id": r[0], "rule": r[1], "source_count": r[2], "source_ids": r[3],
                "confidence": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


# -- Insights --

@app.route("/insight", methods=["POST"])
def insight_create():
    db = get_db()
    data = request.json or {}
    ts = now_iso()
    db["insights"].insert({
        "type": data.get("type", ""),
        "pattern": data.get("pattern", ""),
        "evidence": data.get("evidence", ""),
        "confidence": data.get("confidence", 0.5),
        "valid_until": data.get("valid_until"),
        "created_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/insight/active")
def insight_active():
    db = get_db()
    ts = now_iso()
    rows = db.execute(
        "SELECT id, type, pattern, evidence, confidence, valid_until, created_at FROM insights WHERE valid_until IS NULL OR valid_until > ? ORDER BY confidence DESC",
        [ts]
    ).fetchall()
    results = [{"id": r[0], "type": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "valid_until": r[5], "created_at": r[6]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/insight/list")
def insight_list():
    db = get_db()
    rows = db.execute(
        "SELECT id, type, pattern, evidence, confidence, valid_until, created_at FROM insights ORDER BY created_at DESC"
    ).fetchall()
    results = [{"id": r[0], "type": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "valid_until": r[5], "created_at": r[6]} for r in rows]
    return jsonify({"count": len(results), "results": results})


# -- Proposals --

@app.route("/proposal", methods=["POST"])
def proposal_create():
    db = get_db()
    data = request.json or {}
    ts = now_iso()
    db["proposals"].insert({
        "file_path": data.get("file_path", ""),
        "change_type": data.get("change_type", ""),
        "description": data.get("description", ""),
        "diff_preview": data.get("diff_preview", ""),
        "status": "pending",
        "created_at": ts,
        "resolved_at": None,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/proposal/pending")
def proposal_pending():
    db = get_db()
    rows = db.execute(
        "SELECT id, file_path, change_type, description, diff_preview, status, created_at, resolved_at FROM proposals WHERE status = 'pending' ORDER BY created_at DESC"
    ).fetchall()
    results = [{"id": r[0], "file_path": r[1], "change_type": r[2], "description": r[3],
                "diff_preview": r[4], "status": r[5], "created_at": r[6], "resolved_at": r[7]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/proposal/<int:proposal_id>/approve", methods=["PUT"])
def proposal_approve(proposal_id):
    db = get_db()
    ts = now_iso()
    db.execute("UPDATE proposals SET status = 'approved', resolved_at = ? WHERE id = ?", [ts, proposal_id])
    return jsonify({"status": "approved", "id": proposal_id})


@app.route("/proposal/<int:proposal_id>/reject", methods=["PUT"])
def proposal_reject(proposal_id):
    db = get_db()
    ts = now_iso()
    db.execute("UPDATE proposals SET status = 'rejected', resolved_at = ? WHERE id = ?", [ts, proposal_id])
    return jsonify({"status": "rejected", "id": proposal_id})


if __name__ == "__main__":
    init_db()
    init_embeddings_table()
    app.run(host="0.0.0.0", port=PORT)
