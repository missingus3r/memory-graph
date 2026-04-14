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
  GET  /version
  POST /worldmodel
  GET  /worldmodel/active
  GET  /worldmodel/list?category=
  DELETE /worldmodel/<id>
  POST /goal
  GET  /goal/<id>
  GET  /goal/list?status=&limit=
  PATCH /goal/<id>
  DELETE /goal/<id>
  GET  /goal/active
  GET  /goal/next
  POST /capability
  GET  /capability/list
  GET  /capability/<name>
  PATCH /capability/<name>
  POST /capability/<name>/record  (record run: success/failure/cost/time)
  DELETE /capability/<name>
  GET  /capability/can?domain=...&risk=...
  GET  /autonomy/levels
  POST /autonomy/check
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
VERSION = "1.6.0"
DB_PATH = os.environ.get("FRIDAY_DB_PATH", str(Path.home() / ".friday" / "memory.db"))
PORT = int(os.environ.get("FRIDAY_MEMORY_PORT", "7777"))


def safe_int(val, default=50, min_val=1, max_val=1000):
    """Safely parse an integer with bounds clamping."""
    try:
        v = int(val)
        return max(min_val, min(v, max_val))
    except (TypeError, ValueError):
        return default


def clamp_float(val, default=0.5, lo=0.0, hi=1.0):
    """Safely parse a float and clamp to [lo, hi]."""
    try:
        v = float(val)
        return max(lo, min(v, hi))
    except (TypeError, ValueError):
        return default


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
        )""",
        """CREATE TABLE IF NOT EXISTS importance_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT UNIQUE, score REAL, hits INTEGER DEFAULT 0,
            created_at TEXT, updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS world_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT, pattern TEXT, evidence TEXT,
            confidence REAL DEFAULT 0.5, occurrences INTEGER DEFAULT 1,
            first_seen TEXT, last_seen TEXT, expires_at TEXT,
            created_at TEXT, updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS capabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            domain TEXT,
            description TEXT,
            confidence REAL DEFAULT 0.5,
            success_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            cost_avg REAL DEFAULT 0.0,
            time_avg_sec REAL DEFAULT 0.0,
            error_types TEXT,
            supervision_needed INTEGER DEFAULT 1,
            autonomy_max INTEGER DEFAULT 0,
            max_risk_tier TEXT DEFAULT 'low',
            last_evaluated TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS autonomy_levels (
            level INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            max_risk_tier TEXT,
            requires_checkpoint INTEGER DEFAULT 0,
            requires_rollback INTEGER DEFAULT 0,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            goal_id TEXT UNIQUE,
            title TEXT,
            description TEXT,
            utility REAL DEFAULT 0.5,
            deadline TEXT,
            constraints TEXT,
            success_criteria TEXT,
            subgoals TEXT,
            risk_tier TEXT DEFAULT 'low',
            status TEXT DEFAULT 'active',
            cost_estimated TEXT,
            evidence TEXT,
            progress REAL DEFAULT 0.0,
            autonomy_level INTEGER DEFAULT 0,
            parent_goal TEXT,
            created_at TEXT,
            updated_at TEXT
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
_keywords_cache = {}
_keywords_cache_ts = 0
_keywords_lock = threading.Lock()

_DEFAULT_KEYWORDS = {
    "nota ": 1.0, "nota:": 1.0, "save": 1.0, "guarda": 1.0,
    "remember": 1.0, "recorda": 1.0, "recordar": 1.0, "importante": 1.0,
    "proyecto": 0.8, "project": 0.8, "compra": 0.8, "pagina": 0.8,
    "deploy": 0.8, "push": 0.8, "pushear": 0.8, "commit": 0.8, "notion": 0.8,
    "busca": 0.6, "search": 0.6, "investiga": 0.6, "agrega": 0.6, "crea": 0.6,
}


def _seed_keywords():
    """Seed importance_keywords table with defaults if empty."""
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM importance_keywords").fetchone()[0]
    if count == 0:
        ts = now_iso()
        for kw, score in _DEFAULT_KEYWORDS.items():
            conn.execute(
                "INSERT OR IGNORE INTO importance_keywords (keyword, score, hits, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
                [kw, score, ts, ts])
        conn.commit()
    conn.close()


def _load_keywords():
    """Load keywords from DB with a 60s cache."""
    global _keywords_cache, _keywords_cache_ts
    import time
    now = time.time()
    with _keywords_lock:
        if _keywords_cache and (now - _keywords_cache_ts) < 60:
            return _keywords_cache
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute("SELECT keyword, score FROM importance_keywords ORDER BY score DESC").fetchall()
            conn.close()
            _keywords_cache = {r[0]: r[1] for r in rows}
            _keywords_cache_ts = now
        except Exception:
            _keywords_cache = _DEFAULT_KEYWORDS
        return _keywords_cache


def classify_importance(role, content):
    """Classify message importance (0.0-1.0) based on dynamic keywords from DB."""
    lower = content.lower()
    keywords = _load_keywords()

    # Check keywords by score (highest first, dict is ordered by score DESC)
    best_score = 0.0
    matched_kw = None
    for kw, score in keywords.items():
        if kw in lower and score > best_score:
            best_score = score
            matched_kw = kw

    if matched_kw:
        # Track hit count with thread-safe lock
        with _keywords_lock:
            try:
                conn = sqlite3.connect(DB_PATH)
                conn.execute("UPDATE importance_keywords SET hits = hits + 1, updated_at = ? WHERE keyword = ?",
                             [now_iso(), matched_kw])
                conn.commit()
                conn.close()
            except Exception:
                pass
        return best_score

    # URL detection (medium)
    if _URL_RE.search(lower):
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
            conn.execute("INSERT OR REPLACE INTO embeddings (source_type, source_id, embedding, text_preview, created_at) VALUES (?, ?, ?, ?, ?)",
                         [source_type, source_id, pack_embedding(emb), text[:200], now_iso()])
            conn.commit()
            conn.close()
    threading.Thread(target=_work, daemon=True).start()


def init_embeddings_table():
    """Create embeddings table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT, source_id INTEGER, embedding BLOB, text_preview TEXT, created_at TEXT)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_emb_source ON embeddings (source_type, source_id)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_source ON embeddings(source_type, source_id)")
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
    limit = safe_int(request.args.get("limit", "20"), default=20)
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
    limit = safe_int(request.args.get("limit", "50"), default=50)
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
    days_old = safe_int(data.get("days_old", 7), default=7, min_val=1, max_val=365)

    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_old)).isoformat()

    # Get old entries grouped by ISO week (exclude existing summaries)
    rows = db.execute(
        "SELECT id, timestamp, role, content FROM conversations WHERE timestamp < ? AND role != 'summary' ORDER BY timestamp",
        [cutoff]
    ).fetchall()

    if not rows:
        return jsonify({"status": "ok", "message": "nothing to summarize", "summaries_created": 0})

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
    oldest = db.execute("SELECT COALESCE(MIN(timestamp), '') FROM conversations").fetchone()[0]
    newest = db.execute("SELECT COALESCE(MAX(timestamp), '') FROM conversations").fetchone()[0]

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
    limit = safe_int(request.args.get("limit", "10"), default=10)

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
    limit = safe_int(request.args.get("limit", "50"), default=50)

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
    db.execute("DELETE FROM embeddings WHERE source_type = 'memory' AND source_id = ?", [mem_id])
    return jsonify({"status": "deleted", "id": mem_id})


@app.route("/memory/recall")
def memory_recall():
    db = get_db()
    topic = request.args.get("topic", "")
    limit = safe_int(request.args.get("limit", "10"), default=10)

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
    limit = safe_int(request.args.get("limit", "20"), default=20)

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


@app.route("/entity/<int:entity_id>", methods=["GET"])
def entity_get(entity_id):
    db = get_db()
    rows = db.execute("SELECT id, name, type, details, created_at, updated_at FROM entities WHERE id = ?", [entity_id]).fetchall()
    if not rows:
        return jsonify({"error": "Entity not found"}), 404
    r = rows[0]
    return jsonify({"id": r[0], "name": r[1], "type": r[2], "details": r[3], "created_at": r[4], "updated_at": r[5]})


@app.route("/entity/<int:id>", methods=["DELETE"])
def entity_delete(id):
    db = get_db()
    existing = db.execute("SELECT id FROM entities WHERE id = ?", [id]).fetchone()
    if not existing:
        return jsonify({"error": f"Entity {id} not found"}), 404
    db.execute("DELETE FROM entities WHERE id = ?", [id])
    return jsonify({"status": "deleted", "id": id})


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
    limit = safe_int(request.args.get("limit", "10"), default=10)
    source_type = request.args.get("type", "")

    if not q:
        return jsonify({"error": "query required"}), 400

    query_emb = generate_embedding(q)
    if not query_emb:
        return jsonify({"error": "embedding generation failed"}), 500

    conn = sqlite3.connect(DB_PATH)
    try:
        sql = "SELECT id, source_type, source_id, embedding, text_preview, created_at FROM embeddings"
        params = []
        if source_type:
            sql += " WHERE source_type = ?"
            params.append(source_type)
        sql += " ORDER BY rowid DESC LIMIT 5000"

        rows = conn.execute(sql, params).fetchall()
    finally:
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
    limit = safe_int(request.args.get("limit", "10"), default=10)

    if not q:
        return jsonify({"error": "query required"}), 400

    # FTS results
    conn = sqlite3.connect(DB_PATH)
    try:
        fts_rows = conn.execute(
            "SELECT c.id, c.content, c.role, c.channel, c.timestamp, COALESCE(c.importance, 0.4) FROM conversations c JOIN conversations_fts f ON c.id = f.rowid WHERE conversations_fts MATCH ? ORDER BY rank LIMIT ?",
            [q, limit * 2]
        ).fetchall()

        fts_results = {}
        for i, row in enumerate(fts_rows):
            fts_results[row[0]] = {"id": row[0], "content": row[1], "role": row[2], "channel": row[3], "timestamp": row[4], "importance": row[5], "fts_rank": i + 1}

        # Semantic results — pre-filter to FTS candidate IDs + LIMIT 5000 fallback
        query_emb = generate_embedding(q)
        sem_results = {}
        if query_emb:
            fts_ids = list(fts_results.keys())
            if fts_ids:
                placeholders = ",".join("?" for _ in fts_ids)
                emb_rows = conn.execute(
                    f"SELECT source_id, embedding, text_preview FROM embeddings WHERE source_type = 'conversation' AND source_id IN ({placeholders})",
                    fts_ids
                ).fetchall()
                # Also load recent embeddings not in FTS results as fallback
                emb_rows += conn.execute(
                    f"SELECT source_id, embedding, text_preview FROM embeddings WHERE source_type = 'conversation' AND source_id NOT IN ({placeholders}) ORDER BY rowid DESC LIMIT 5000",
                    fts_ids
                ).fetchall()
            else:
                emb_rows = conn.execute(
                    "SELECT source_id, embedding, text_preview FROM embeddings WHERE source_type = 'conversation' ORDER BY rowid DESC LIMIT 5000"
                ).fetchall()
            scored = []
            for row in emb_rows:
                emb = unpack_embedding(row[1])
                score = cosine_similarity(query_emb, emb)
                scored.append((row[0], score, row[2]))
            scored.sort(key=lambda x: x[1], reverse=True)
            for i, (sid, score, preview) in enumerate(scored[:limit * 2]):
                sem_results[sid] = {"source_id": sid, "score": score, "sem_rank": i + 1, "text_preview": preview}
    finally:
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
        # Index missing conversations
        rows = conn.execute("SELECT id, content FROM conversations WHERE id NOT IN (SELECT source_id FROM embeddings WHERE source_type=?)", ["conversation"]).fetchall()
        for row in rows:
            emb = generate_embedding(row[1])
            if emb:
                conn.execute("INSERT OR REPLACE INTO embeddings (source_type, source_id, embedding, text_preview, created_at) VALUES (?, ?, ?, ?, ?)",
                             ["conversation", row[0], pack_embedding(emb), row[1][:200], now_iso()])
                conn.commit()
        # Index missing memories
        rows = conn.execute("SELECT id, name, description, content FROM memories WHERE id NOT IN (SELECT source_id FROM embeddings WHERE source_type=?)", ["memory"]).fetchall()
        for row in rows:
            text = f"{row[1]} {row[2]} {row[3]}"
            emb = generate_embedding(text)
            if emb:
                conn.execute("INSERT OR REPLACE INTO embeddings (source_type, source_id, embedding, text_preview, created_at) VALUES (?, ?, ?, ?, ?)",
                             ["memory", row[0], pack_embedding(emb), text[:200], now_iso()])
                conn.commit()
        conn.close()

    threading.Thread(target=_reindex, daemon=True).start()
    return jsonify({"status": "ok", "message": "reindexing started"})


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
    return jsonify({"error": "Graph not found"}), 404


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
            try:
                return jsonify(json.loads(row[0]))
            except (json.JSONDecodeError, TypeError):
                return jsonify({"value": row[0]})
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
    limit = safe_int(request.args.get("limit", "5"), default=5)
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
    limit = safe_int(request.args.get("limit", "100"), default=100)
    rows = db.execute(
        "SELECT id, date, analysis, insights, actions, created_at FROM reflections ORDER BY created_at DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "date": r[1], "analysis": r[2], "insights": r[3],
                "actions": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/reflection/<int:reflection_id>", methods=["GET"])
def reflection_get(reflection_id):
    db = get_db()
    rows = db.execute("SELECT id, date, analysis, insights, actions, created_at FROM reflections WHERE id = ?", [reflection_id]).fetchall()
    if not rows:
        return jsonify({"error": "Reflection not found"}), 404
    r = rows[0]
    return jsonify({"id": r[0], "date": r[1], "analysis": r[2], "insights": r[3], "actions": r[4], "created_at": r[5]})


@app.route("/reflection/<int:id>", methods=["DELETE"])
def reflection_delete(id):
    db = get_db()
    existing = db.execute("SELECT id FROM reflections WHERE id = ?", [id]).fetchone()
    if not existing:
        return jsonify({"error": f"Reflection {id} not found"}), 404
    db.execute("DELETE FROM reflections WHERE id = ?", [id])
    return jsonify({"status": "deleted", "id": id})


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
    limit = safe_int(request.args.get("limit", "100"), default=100)
    rows = db.execute(
        "SELECT id, name, trigger_pattern, description, steps, times_used, last_used, created_at FROM skills ORDER BY times_used DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "name": r[1], "trigger_pattern": r[2], "description": r[3],
                "steps": r[4], "times_used": r[5], "last_used": r[6], "created_at": r[7]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/skill/<int:skill_id>", methods=["GET"])
def skill_get(skill_id):
    db = get_db()
    rows = db.execute("SELECT id, name, trigger_pattern, description, steps, times_used, last_used, created_at FROM skills WHERE id = ?", [skill_id]).fetchall()
    if not rows:
        return jsonify({"error": "Skill not found"}), 404
    r = rows[0]
    return jsonify({"id": r[0], "name": r[1], "trigger_pattern": r[2], "description": r[3], "steps": r[4], "times_used": r[5], "last_used": r[6], "created_at": r[7]})


@app.route("/skill/<int:id>", methods=["DELETE"])
def skill_delete(id):
    db = get_db()
    existing = db.execute("SELECT id FROM skills WHERE id = ?", [id]).fetchone()
    if not existing:
        return jsonify({"error": f"Skill {id} not found"}), 404
    db.execute("DELETE FROM skills WHERE id = ?", [id])
    return jsonify({"status": "deleted", "id": id})


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
        "confidence": clamp_float(data.get("confidence", 0.5)),
        "created_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/preference/list")
def preference_list():
    db = get_db()
    limit = safe_int(request.args.get("limit", "100"), default=100)
    rows = db.execute(
        "SELECT id, rule, source_count, source_ids, confidence, created_at FROM preferences ORDER BY confidence DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "rule": r[1], "source_count": r[2], "source_ids": r[3],
                "confidence": r[4], "created_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/preference/<int:id>", methods=["DELETE"])
def preference_delete(id):
    db = get_db()
    existing = db.execute("SELECT id FROM preferences WHERE id = ?", [id]).fetchone()
    if not existing:
        return jsonify({"error": f"Preference {id} not found"}), 404
    db.execute("DELETE FROM preferences WHERE id = ?", [id])
    return jsonify({"status": "deleted", "id": id})


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
        "confidence": clamp_float(data.get("confidence", 0.5)),
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
    limit = safe_int(request.args.get("limit", "100"), default=100)
    rows = db.execute(
        "SELECT id, type, pattern, evidence, confidence, valid_until, created_at FROM insights ORDER BY created_at DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "type": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "valid_until": r[5], "created_at": r[6]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/insight/<int:id>", methods=["DELETE"])
def insight_delete(id):
    db = get_db()
    existing = db.execute("SELECT id FROM insights WHERE id = ?", [id]).fetchone()
    if not existing:
        return jsonify({"error": f"Insight {id} not found"}), 404
    db.execute("DELETE FROM insights WHERE id = ?", [id])
    return jsonify({"status": "deleted", "id": id})


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


@app.route("/proposal/<int:proposal_id>", methods=["GET"])
def proposal_get(proposal_id):
    db = get_db()
    rows = db.execute("SELECT id, file_path, change_type, description, diff_preview, status, created_at, resolved_at FROM proposals WHERE id = ?", [proposal_id]).fetchall()
    if not rows:
        return jsonify({"error": "Proposal not found"}), 404
    r = rows[0]
    return jsonify({"id": r[0], "file_path": r[1], "change_type": r[2], "description": r[3], "diff_preview": r[4], "status": r[5], "created_at": r[6], "resolved_at": r[7]})


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


# -- World Model --

@app.route("/worldmodel", methods=["POST"])
def worldmodel_create():
    """Create or update a world model entry. If a matching pattern+category exists, update it."""
    db = get_db()
    data = request.json or {}
    ts = now_iso()
    category = data.get("category", "behavior")
    pattern = data.get("pattern", "")

    # Check if this pattern already exists in this category
    existing = db.execute(
        "SELECT id, occurrences, confidence FROM world_model WHERE category = ? AND pattern = ?",
        [category, pattern]
    ).fetchone()

    if existing:
        new_occ = existing[1] + 1
        # Decay-aware confidence: boost more if recently seen
        try:
            last_seen_dt = datetime.datetime.fromisoformat(
                db.execute("SELECT last_seen FROM world_model WHERE id = ?", [existing[0]]).fetchone()[0]
            )
            days_since = (datetime.datetime.now(datetime.timezone.utc) - last_seen_dt).days
        except Exception:
            days_since = 30
        if days_since < 7:
            new_conf = min(0.99, existing[2] + 0.03)
        else:
            new_conf = min(0.99, existing[2] + 0.01)
        db.execute(
            "UPDATE world_model SET occurrences = ?, confidence = ?, last_seen = ?, updated_at = ?, evidence = COALESCE(?, evidence), expires_at = COALESCE(?, expires_at) WHERE id = ?",
            [new_occ, new_conf, ts, ts, data.get("evidence"), data.get("expires_at"), existing[0]]
        )
        return jsonify({"status": "updated", "id": existing[0], "occurrences": new_occ, "confidence": new_conf})

    db["world_model"].insert({
        "category": category,
        "pattern": pattern,
        "evidence": data.get("evidence", ""),
        "confidence": clamp_float(data.get("confidence", 0.5)),
        "occurrences": 1,
        "first_seen": ts,
        "last_seen": ts,
        "expires_at": data.get("expires_at", ""),
        "created_at": ts,
        "updated_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "created", "id": last_id})


@app.route("/worldmodel/active")
def worldmodel_active():
    """Get active (non-expired) world model entries with confidence >= 0.4."""
    db = get_db()
    ts = now_iso()
    rows = db.execute(
        "SELECT id, category, pattern, evidence, confidence, occurrences, first_seen, last_seen, expires_at, created_at FROM world_model WHERE confidence >= 0.4 AND (expires_at = '' OR expires_at IS NULL OR expires_at > ?) ORDER BY confidence DESC, occurrences DESC",
        [ts]
    ).fetchall()
    results = [{"id": r[0], "category": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "occurrences": r[5], "first_seen": r[6],
                "last_seen": r[7], "expires_at": r[8], "created_at": r[9]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/worldmodel/list")
def worldmodel_list():
    """List all world model entries."""
    db = get_db()
    category = request.args.get("category", "")
    limit = safe_int(request.args.get("limit", "100"), default=100)
    if category:
        rows = db.execute(
            "SELECT id, category, pattern, evidence, confidence, occurrences, first_seen, last_seen, expires_at, created_at FROM world_model WHERE category = ? ORDER BY confidence DESC LIMIT ?",
            [category, limit]
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, category, pattern, evidence, confidence, occurrences, first_seen, last_seen, expires_at, created_at FROM world_model ORDER BY category, confidence DESC LIMIT ?",
            [limit]
        ).fetchall()
    results = [{"id": r[0], "category": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "occurrences": r[5], "first_seen": r[6],
                "last_seen": r[7], "expires_at": r[8], "created_at": r[9]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/worldmodel/<int:entry_id>", methods=["DELETE"])
def worldmodel_delete(entry_id):
    db = get_db()
    existing = db.execute("SELECT id FROM world_model WHERE id = ?", [entry_id]).fetchone()
    if not existing:
        return jsonify({"error": f"World model entry {entry_id} not found"}), 404
    db.execute("DELETE FROM world_model WHERE id = ?", [entry_id])
    return jsonify({"status": "deleted", "id": entry_id})


# -- Importance Keywords (dynamic) --

@app.route("/keywords", methods=["GET"])
def keywords_list():
    """List all importance keywords with scores and hit counts."""
    db = get_db()
    rows = db.execute(
        "SELECT id, keyword, score, hits, created_at, updated_at FROM importance_keywords ORDER BY score DESC, hits DESC"
    ).fetchall()
    results = [{"id": r[0], "keyword": r[1], "score": r[2], "hits": r[3],
                "created_at": r[4], "updated_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/keywords", methods=["POST"])
def keywords_create():
    """Add or update a keyword. {keyword, score}"""
    db = get_db()
    data = request.json or {}
    keyword = data.get("keyword", "").lower().strip()
    score = clamp_float(data.get("score", 0.6), default=0.6)
    ts = now_iso()
    existing = db.execute("SELECT id FROM importance_keywords WHERE keyword = ?", [keyword]).fetchone()
    if existing:
        db.execute("UPDATE importance_keywords SET score = ?, updated_at = ? WHERE keyword = ?", [score, ts, keyword])
        with _keywords_lock:
            global _keywords_cache_ts
            _keywords_cache_ts = 0  # invalidate cache
        return jsonify({"status": "updated", "keyword": keyword, "score": score})
    db.execute(
        "INSERT INTO importance_keywords (keyword, score, hits, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
        [keyword, score, ts, ts])
    with _keywords_lock:
        _keywords_cache_ts = 0
    return jsonify({"status": "created", "keyword": keyword, "score": score})


@app.route("/keywords/<int:kw_id>", methods=["DELETE"])
def keywords_delete(kw_id):
    db = get_db()
    existing = db.execute("SELECT id FROM importance_keywords WHERE id = ?", [kw_id]).fetchone()
    if not existing:
        return jsonify({"error": f"Keyword {kw_id} not found"}), 404
    db.execute("DELETE FROM importance_keywords WHERE id = ?", [kw_id])
    with _keywords_lock:
        global _keywords_cache_ts
        _keywords_cache_ts = 0
    return jsonify({"status": "deleted", "id": kw_id})


# -- Goal Engine --
import uuid


def _goal_row_to_dict(r):
    def _jload(s, default):
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default
    return {
        "id": r[0], "goal_id": r[1], "title": r[2], "description": r[3],
        "utility": r[4], "deadline": r[5],
        "constraints": _jload(r[6], []),
        "success_criteria": _jload(r[7], []),
        "subgoals": _jload(r[8], []),
        "risk_tier": r[9], "status": r[10], "cost_estimated": r[11],
        "evidence": _jload(r[12], []),
        "progress": r[13], "autonomy_level": r[14], "parent_goal": r[15],
        "created_at": r[16], "updated_at": r[17],
    }


_GOAL_COLUMNS = "id, goal_id, title, description, utility, deadline, constraints, success_criteria, subgoals, risk_tier, status, cost_estimated, evidence, progress, autonomy_level, parent_goal, created_at, updated_at"


@app.route("/goal", methods=["POST"])
def goal_create():
    """Create a new goal. Body: {title, description?, utility?, deadline?, constraints?, success_criteria?, subgoals?, risk_tier?, cost_estimated?, autonomy_level?, parent_goal?}"""
    db = get_db()
    data = request.json or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title required"}), 400

    def _jdump(v):
        if v is None:
            return None
        if isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False)

    ts = now_iso()
    goal_id = data.get("goal_id") or f"g_{uuid.uuid4().hex[:8]}"
    row = {
        "goal_id": goal_id,
        "title": title,
        "description": data.get("description", ""),
        "utility": clamp_float(data.get("utility", 0.5), default=0.5),
        "deadline": data.get("deadline", ""),
        "constraints": _jdump(data.get("constraints", [])),
        "success_criteria": _jdump(data.get("success_criteria", [])),
        "subgoals": _jdump(data.get("subgoals", [])),
        "risk_tier": data.get("risk_tier", "low"),
        "status": data.get("status", "active"),
        "cost_estimated": data.get("cost_estimated", ""),
        "evidence": _jdump(data.get("evidence", [])),
        "progress": clamp_float(data.get("progress", 0.0), default=0.0),
        "autonomy_level": safe_int(data.get("autonomy_level", 0), default=0, min_val=0, max_val=5),
        "parent_goal": data.get("parent_goal", ""),
        "created_at": ts,
        "updated_at": ts,
    }
    try:
        db["goals"].insert(row)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    created = db.execute(f"SELECT {_GOAL_COLUMNS} FROM goals WHERE goal_id = ?", [goal_id]).fetchone()
    return jsonify({"status": "created", "goal": _goal_row_to_dict(created)})


@app.route("/goal/<goal_id>", methods=["GET"])
def goal_get(goal_id):
    db = get_db()
    row = db.execute(f"SELECT {_GOAL_COLUMNS} FROM goals WHERE goal_id = ?", [goal_id]).fetchone()
    if not row:
        return jsonify({"error": f"Goal {goal_id} not found"}), 404
    return jsonify({"goal": _goal_row_to_dict(row)})


@app.route("/goal/list", methods=["GET"])
def goal_list():
    db = get_db()
    status = request.args.get("status", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    if status:
        rows = db.execute(
            f"SELECT {_GOAL_COLUMNS} FROM goals WHERE status = ? ORDER BY utility DESC, deadline ASC LIMIT ?",
            [status, limit]).fetchall()
    else:
        rows = db.execute(
            f"SELECT {_GOAL_COLUMNS} FROM goals ORDER BY status, utility DESC LIMIT ?",
            [limit]).fetchall()
    results = [_goal_row_to_dict(r) for r in rows]
    return jsonify({"count": len(results), "goals": results})


@app.route("/goal/active", methods=["GET"])
def goal_active():
    db = get_db()
    rows = db.execute(
        f"SELECT {_GOAL_COLUMNS} FROM goals WHERE status = 'active' ORDER BY utility DESC, deadline ASC"
    ).fetchall()
    results = [_goal_row_to_dict(r) for r in rows]
    return jsonify({"count": len(results), "goals": results})


@app.route("/goal/next", methods=["GET"])
def goal_next():
    """Return the next recommended goal by priority = utility * urgency.
    Urgency = 1.0 if no deadline, else clamp(1 / days_remaining, 0, 3)."""
    db = get_db()
    rows = db.execute(
        f"SELECT {_GOAL_COLUMNS} FROM goals WHERE status = 'active'"
    ).fetchall()
    if not rows:
        return jsonify({"goal": None, "reason": "no active goals"})

    now = datetime.datetime.now(datetime.timezone.utc)
    scored = []
    for r in rows:
        g_dict = _goal_row_to_dict(r)
        deadline = g_dict.get("deadline") or ""
        urgency = 1.0
        if deadline:
            try:
                dl = datetime.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                if dl.tzinfo is None:
                    dl = dl.replace(tzinfo=datetime.timezone.utc)
                days = max((dl - now).total_seconds() / 86400.0, 0.01)
                urgency = max(0.1, min(3.0, 7.0 / days))
            except Exception:
                urgency = 1.0
        priority = g_dict["utility"] * urgency * (1.0 - g_dict.get("progress", 0.0))
        scored.append((priority, g_dict, urgency))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0]
    return jsonify({"goal": best[1], "priority": round(best[0], 4), "urgency": round(best[2], 4)})


@app.route("/goal/<goal_id>", methods=["PATCH"])
def goal_update(goal_id):
    db = get_db()
    existing = db.execute("SELECT id FROM goals WHERE goal_id = ?", [goal_id]).fetchone()
    if not existing:
        return jsonify({"error": f"Goal {goal_id} not found"}), 404
    data = request.json or {}
    updatable = {
        "title", "description", "utility", "deadline", "constraints",
        "success_criteria", "subgoals", "risk_tier", "status",
        "cost_estimated", "evidence", "progress", "autonomy_level", "parent_goal",
    }
    json_fields = {"constraints", "success_criteria", "subgoals", "evidence"}
    sets = []
    vals = []
    for k, v in data.items():
        if k not in updatable:
            continue
        if k in json_fields and not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        if k == "utility":
            v = clamp_float(v, default=0.5)
        if k == "progress":
            v = clamp_float(v, default=0.0)
        if k == "autonomy_level":
            v = safe_int(v, default=0, min_val=0, max_val=5)
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return jsonify({"error": "no updatable fields provided"}), 400
    sets.append("updated_at = ?")
    vals.append(now_iso())
    vals.append(goal_id)
    db.execute(f"UPDATE goals SET {', '.join(sets)} WHERE goal_id = ?", vals)
    row = db.execute(f"SELECT {_GOAL_COLUMNS} FROM goals WHERE goal_id = ?", [goal_id]).fetchone()
    return jsonify({"status": "updated", "goal": _goal_row_to_dict(row)})


@app.route("/goal/<goal_id>", methods=["DELETE"])
def goal_delete(goal_id):
    db = get_db()
    existing = db.execute("SELECT id FROM goals WHERE goal_id = ?", [goal_id]).fetchone()
    if not existing:
        return jsonify({"error": f"Goal {goal_id} not found"}), 404
    db.execute("DELETE FROM goals WHERE goal_id = ?", [goal_id])
    return jsonify({"status": "deleted", "goal_id": goal_id})


# -- Autonomy Levels --

_AUTONOMY_LEVELS = [
    (0, "Suggest Only", "Sólo sugerir; ninguna ejecución real.", "any", 0, 0),
    (1, "Sandbox", "Ejecuta únicamente en sandbox/dry-run.", "low", 0, 1),
    (2, "Low-Risk Act", "Puede actuar en sistemas de bajo riesgo (lectura, logs, memoria propia).", "low", 0, 1),
    (3, "Bounded Act", "Actúa con límites monetarios/temporales; cualquier coste requiere budget.", "medium", 1, 1),
    (4, "Long Chain", "Cadenas largas con checkpoints; pausa ante señales inesperadas.", "medium", 1, 1),
    (5, "Self-Modify", "Puede auto-modificar prompts/policies/skills en ramas aisladas con rollback obligatorio.", "high", 1, 1),
]


def _seed_autonomy_levels():
    conn = sqlite3.connect(DB_PATH)
    ts = now_iso()
    for lvl, name, desc, risk, ckpt, rb in _AUTONOMY_LEVELS:
        conn.execute(
            "INSERT OR IGNORE INTO autonomy_levels (level, name, description, max_risk_tier, requires_checkpoint, requires_rollback, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [lvl, name, desc, risk, ckpt, rb, ts])
    conn.commit()
    conn.close()


@app.route("/autonomy/levels", methods=["GET"])
def autonomy_levels_list():
    db = get_db()
    rows = db.execute(
        "SELECT level, name, description, max_risk_tier, requires_checkpoint, requires_rollback FROM autonomy_levels ORDER BY level"
    ).fetchall()
    results = [{"level": r[0], "name": r[1], "description": r[2], "max_risk_tier": r[3],
                "requires_checkpoint": bool(r[4]), "requires_rollback": bool(r[5])} for r in rows]
    return jsonify({"count": len(results), "levels": results})


_RISK_ORDER = {"any": 0, "low": 1, "medium": 2, "high": 3}


@app.route("/autonomy/check", methods=["POST"])
def autonomy_check():
    """Decide if an action is allowed. Body: {capability?, domain?, proposed_level, risk_tier}.
    Returns {allowed: bool, reason, required_supervision, capability_autonomy_max}."""
    db = get_db()
    data = request.json or {}
    cap_name = data.get("capability") or data.get("domain") or ""
    proposed_level = safe_int(data.get("proposed_level", 0), default=0, min_val=0, max_val=5)
    risk_tier = (data.get("risk_tier") or "low").lower()

    lvl_row = db.execute(
        "SELECT level, max_risk_tier, requires_checkpoint FROM autonomy_levels WHERE level = ?",
        [proposed_level]).fetchone()
    if not lvl_row:
        return jsonify({"allowed": False, "reason": f"unknown autonomy level {proposed_level}"}), 400

    level_risk = lvl_row[1]
    risk_ok = _RISK_ORDER.get(risk_tier, 99) <= _RISK_ORDER.get(level_risk, 0)

    cap_row = None
    if cap_name:
        cap_row = db.execute(
            "SELECT name, confidence, autonomy_max, supervision_needed, max_risk_tier FROM capabilities WHERE name = ? OR domain = ?",
            [cap_name, cap_name]).fetchone()

    cap_ok = True
    cap_reason = None
    cap_autonomy_max = 5
    supervision = False
    if cap_row:
        cap_autonomy_max = cap_row[2] or 0
        supervision = bool(cap_row[3])
        cap_ok = proposed_level <= cap_autonomy_max
        if not cap_ok:
            cap_reason = f"capability '{cap_row[0]}' is capped at autonomy L{cap_autonomy_max}"

    allowed = risk_ok and cap_ok
    reason = cap_reason or (None if allowed else f"risk {risk_tier} exceeds level L{proposed_level} max ({level_risk})")
    return jsonify({
        "allowed": allowed,
        "reason": reason or "ok",
        "required_supervision": supervision or proposed_level <= 1,
        "capability_autonomy_max": cap_autonomy_max,
        "proposed_level": proposed_level,
        "risk_tier": risk_tier,
    })


# -- Capabilities --

_DEFAULT_CAPABILITIES = [
    ("coding", "engineering", "Write, edit, and debug code.", 0.65, 2, "medium"),
    ("research", "information", "Gather and synthesize information from multiple sources.", 0.7, 2, "low"),
    ("email_drafting", "communication", "Draft email messages for the user's review.", 0.6, 1, "low"),
    ("scheduling", "planning", "Create and manage calendar events and crons.", 0.65, 2, "low"),
    ("long_horizon_planning", "planning", "Decompose ambiguous goals into plans spanning days/weeks.", 0.4, 1, "medium"),
    ("memory_recall", "memory", "Retrieve relevant context from the memory system.", 0.75, 3, "low"),
    ("fact_reliability", "verification", "Judge whether a claim is supported by evidence.", 0.5, 1, "medium"),
    ("multimodal_work", "perception", "Understand and act on images, audio, and mixed media.", 0.55, 1, "low"),
    ("tool_use", "execution", "Select and invoke the right tool for a task.", 0.7, 2, "low"),
    ("self_modification", "meta", "Propose changes to own prompts, skills, or code.", 0.3, 0, "high"),
]


def _seed_capabilities():
    conn = sqlite3.connect(DB_PATH)
    ts = now_iso()
    for name, domain, desc, conf, autonomy_max, risk in _DEFAULT_CAPABILITIES:
        conn.execute(
            """INSERT OR IGNORE INTO capabilities
               (name, domain, description, confidence, success_count, failure_count, cost_avg, time_avg_sec, error_types, supervision_needed, autonomy_max, max_risk_tier, last_evaluated, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, 0, 0.0, 0.0, '[]', 1, ?, ?, ?, ?, ?)""",
            [name, domain, desc, conf, autonomy_max, risk, ts, ts, ts])
    conn.commit()
    conn.close()


_CAP_COLUMNS = "id, name, domain, description, confidence, success_count, failure_count, cost_avg, time_avg_sec, error_types, supervision_needed, autonomy_max, max_risk_tier, last_evaluated, created_at, updated_at"


def _cap_row_to_dict(r):
    try:
        errors = json.loads(r[9]) if r[9] else []
    except Exception:
        errors = []
    total = (r[5] or 0) + (r[6] or 0)
    success_rate = (r[5] / total) if total > 0 else None
    return {
        "id": r[0], "name": r[1], "domain": r[2], "description": r[3],
        "confidence": r[4], "success_count": r[5], "failure_count": r[6],
        "cost_avg": r[7], "time_avg_sec": r[8], "error_types": errors,
        "supervision_needed": bool(r[10]), "autonomy_max": r[11],
        "max_risk_tier": r[12], "last_evaluated": r[13],
        "created_at": r[14], "updated_at": r[15],
        "success_rate": success_rate,
        "runs": total,
    }


@app.route("/capability", methods=["POST"])
def capability_create():
    db = get_db()
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    ts = now_iso()
    row = {
        "name": name,
        "domain": data.get("domain", ""),
        "description": data.get("description", ""),
        "confidence": clamp_float(data.get("confidence", 0.5), default=0.5),
        "success_count": safe_int(data.get("success_count", 0), default=0, min_val=0, max_val=10**9),
        "failure_count": safe_int(data.get("failure_count", 0), default=0, min_val=0, max_val=10**9),
        "cost_avg": clamp_float(data.get("cost_avg", 0.0), default=0.0, lo=0.0, hi=10**6),
        "time_avg_sec": clamp_float(data.get("time_avg_sec", 0.0), default=0.0, lo=0.0, hi=10**6),
        "error_types": json.dumps(data.get("error_types", []), ensure_ascii=False),
        "supervision_needed": 1 if data.get("supervision_needed", True) else 0,
        "autonomy_max": safe_int(data.get("autonomy_max", 0), default=0, min_val=0, max_val=5),
        "max_risk_tier": data.get("max_risk_tier", "low"),
        "last_evaluated": ts,
        "created_at": ts,
        "updated_at": ts,
    }
    try:
        db["capabilities"].insert(row)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    created = db.execute(f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE name = ?", [name]).fetchone()
    return jsonify({"status": "created", "capability": _cap_row_to_dict(created)})


@app.route("/capability/list", methods=["GET"])
def capability_list():
    db = get_db()
    domain = request.args.get("domain", "")
    limit = safe_int(request.args.get("limit"), default=100, max_val=500)
    if domain:
        rows = db.execute(
            f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE domain = ? ORDER BY confidence DESC LIMIT ?",
            [domain, limit]).fetchall()
    else:
        rows = db.execute(
            f"SELECT {_CAP_COLUMNS} FROM capabilities ORDER BY domain, confidence DESC LIMIT ?",
            [limit]).fetchall()
    results = [_cap_row_to_dict(r) for r in rows]
    return jsonify({"count": len(results), "capabilities": results})


@app.route("/capability/can", methods=["GET"])
def capability_can():
    """Can capability do a task of given risk? Returns decision + calibrated confidence.
    Query: ?domain=<name>&risk=<low|medium|high>"""
    db = get_db()
    name = request.args.get("domain") or request.args.get("name") or ""
    risk = (request.args.get("risk") or "low").lower()
    row = db.execute(
        f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE name = ? OR domain = ? ORDER BY confidence DESC LIMIT 1",
        [name, name]).fetchone()
    if not row:
        return jsonify({"allowed": False, "reason": f"no capability for '{name}'", "confidence": None}), 404
    cap = _cap_row_to_dict(row)
    risk_ok = _RISK_ORDER.get(risk, 99) <= _RISK_ORDER.get(cap["max_risk_tier"], 0)
    conf_threshold = 0.7 if risk == "high" else (0.55 if risk == "medium" else 0.4)
    conf_ok = cap["confidence"] >= conf_threshold
    allowed = risk_ok and conf_ok
    reason = "ok" if allowed else (
        f"risk {risk} exceeds capability max ({cap['max_risk_tier']})" if not risk_ok
        else f"confidence {cap['confidence']:.2f} below threshold {conf_threshold} for risk {risk}"
    )
    return jsonify({
        "allowed": allowed,
        "reason": reason,
        "confidence": cap["confidence"],
        "success_rate": cap["success_rate"],
        "autonomy_max": cap["autonomy_max"],
        "supervision_needed": cap["supervision_needed"],
    })


@app.route("/capability/<name>", methods=["GET"])
def capability_get(name):
    db = get_db()
    row = db.execute(f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE name = ?", [name]).fetchone()
    if not row:
        return jsonify({"error": f"Capability '{name}' not found"}), 404
    return jsonify({"capability": _cap_row_to_dict(row)})


@app.route("/capability/<name>", methods=["PATCH"])
def capability_update(name):
    db = get_db()
    existing = db.execute("SELECT id FROM capabilities WHERE name = ?", [name]).fetchone()
    if not existing:
        return jsonify({"error": f"Capability '{name}' not found"}), 404
    data = request.json or {}
    updatable = {"domain", "description", "confidence", "cost_avg", "time_avg_sec",
                 "error_types", "supervision_needed", "autonomy_max", "max_risk_tier"}
    sets = []
    vals = []
    for k, v in data.items():
        if k not in updatable:
            continue
        if k == "confidence":
            v = clamp_float(v, default=0.5)
        elif k == "autonomy_max":
            v = safe_int(v, default=0, min_val=0, max_val=5)
        elif k == "supervision_needed":
            v = 1 if v else 0
        elif k == "error_types" and not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return jsonify({"error": "no updatable fields provided"}), 400
    sets.append("updated_at = ?")
    vals.append(now_iso())
    vals.append(name)
    db.execute(f"UPDATE capabilities SET {', '.join(sets)} WHERE name = ?", vals)
    row = db.execute(f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE name = ?", [name]).fetchone()
    return jsonify({"status": "updated", "capability": _cap_row_to_dict(row)})


@app.route("/capability/<name>/record", methods=["POST"])
def capability_record(name):
    """Record a run outcome. Body: {outcome: 'success'|'failure', cost?, time_sec?, error_type?}"""
    db = get_db()
    row = db.execute(f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE name = ?", [name]).fetchone()
    if not row:
        return jsonify({"error": f"Capability '{name}' not found"}), 404
    cap = _cap_row_to_dict(row)
    data = request.json or {}
    outcome = (data.get("outcome") or "").lower()
    if outcome not in ("success", "failure"):
        return jsonify({"error": "outcome must be 'success' or 'failure'"}), 400
    cost = float(data.get("cost", 0.0) or 0.0)
    time_sec = float(data.get("time_sec", 0.0) or 0.0)
    error_type = (data.get("error_type") or "").strip() if outcome == "failure" else None

    new_success = cap["success_count"] + (1 if outcome == "success" else 0)
    new_failure = cap["failure_count"] + (1 if outcome == "failure" else 0)
    total = new_success + new_failure

    # Rolling averages
    prev_total = cap["runs"]
    new_cost_avg = ((cap["cost_avg"] * prev_total) + cost) / total if total else 0.0
    new_time_avg = ((cap["time_avg_sec"] * prev_total) + time_sec) / total if total else 0.0

    # Calibrated confidence: blend prior (0.5) with observed success rate, weighted by runs
    prior = 0.5
    prior_weight = 5.0  # pseudo-count
    observed_rate = new_success / total if total else prior
    new_conf = (prior * prior_weight + observed_rate * total) / (prior_weight + total)

    error_types = cap["error_types"] or []
    if error_type:
        error_types = (error_types + [error_type])[-20:]

    ts = now_iso()
    db.execute(
        """UPDATE capabilities SET success_count = ?, failure_count = ?, cost_avg = ?, time_avg_sec = ?,
           confidence = ?, error_types = ?, last_evaluated = ?, updated_at = ? WHERE name = ?""",
        [new_success, new_failure, new_cost_avg, new_time_avg, new_conf,
         json.dumps(error_types, ensure_ascii=False), ts, ts, name])
    updated = db.execute(f"SELECT {_CAP_COLUMNS} FROM capabilities WHERE name = ?", [name]).fetchone()
    return jsonify({"status": "recorded", "capability": _cap_row_to_dict(updated)})


@app.route("/capability/<name>", methods=["DELETE"])
def capability_delete(name):
    db = get_db()
    existing = db.execute("SELECT id FROM capabilities WHERE name = ?", [name]).fetchone()
    if not existing:
        return jsonify({"error": f"Capability '{name}' not found"}), 404
    db.execute("DELETE FROM capabilities WHERE name = ?", [name])
    return jsonify({"status": "deleted", "name": name})


if __name__ == "__main__":
    init_db()
    init_embeddings_table()
    _seed_keywords()
    _seed_autonomy_levels()
    _seed_capabilities()
    app.run(host="0.0.0.0", port=PORT)
