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
  GET  /memory/episodic?limit=&hours=
  GET  /memory/semantic?type=&limit=
  GET  /memory/procedural?limit=
  POST /memory/<id>/verify
  POST /memory/decay   (apply confidence decay pass)
  POST /plan
  GET  /plan/<plan_id>
  GET  /plan/list?goal_id=
  POST /plan/<plan_id>/node
  PATCH /plan/node/<node_id>
  DELETE /plan/<plan_id>
  POST /wm/entity
  GET  /wm/entity/list
  POST /wm/relation
  GET  /wm/relation/list
  POST /wm/event
  GET  /wm/event/list
  POST /wm/prediction
  GET  /wm/prediction/list
  PATCH /wm/prediction/<id>/resolve
  POST /verify
  GET  /verify/list?subject_type=&subject_id=
  POST /sandbox/execute    (dry-run | simulation | live)
  GET  /sandbox/list
  POST /skill/<id>/record  (record execution outcome)
  PATCH /skill/<id>/promote (maturity: draft→beta→stable)
  POST /experiment
  GET  /experiment/list?status=
  POST /experiment/<id>/variant
  POST /experiment/<id>/observation
  PATCH /experiment/<id>/conclude
  POST /metric
  GET  /metric/list?name=&limit=
  GET  /metric/summary
  POST /worldmodel/<id>/promote  (promote soft observation → wm_event/relation/prediction)
  POST /cron/active              (bulk replace snapshot of currently-scheduled crons)
  GET  /cron/active
  GET  /cron/prompts             (reads ~/.claude/cron-prompts.md, returns parsed sections)

Table ownership notes (see split in CLAUDE.md / Self-Improving Harness):
  - entities        → IDENTITY layer: who/what things ARE (Bruno, gmail,
                      Claude Opus 4.6). Long-lived. Not state.
  - wm_entities     → STATE layer: current state of the world
                      (deploy_status=pending, Bruno_mood=cansado).
                      Mutable, last_verified matters.
  - world_model     → SOFT OBSERVATION INBOX: loose patterns before they
                      earn a structured shape.
  - wm_events /
    wm_relations /
    wm_predictions  → STRUCTURED KNOWLEDGE: graduated from world_model
                      when an observation becomes testable, causal, or
                      subject-predicate-object. Use POST /worldmodel/<id>/promote.
"""

import json
import os
import re
import sqlite3
import datetime
import struct
import math
import threading
import time
import shutil
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from flask import Flask, request, jsonify, g, send_file, session, redirect, make_response
import secrets as _secrets
import sqlite_utils

# ── Config ──
VERSION = "2.14.0"
DB_PATH = os.environ.get("FRIDAY_DB_PATH", str(Path.home() / ".friday" / "memory.db"))
PORT = int(os.environ.get("FRIDAY_MEMORY_PORT", "7777"))

# ── Auth (protects the /graph UI only, not the API endpoints) ──
GRAPH_USER = os.environ.get("FRIDAY_GRAPH_USER", "admin")
GRAPH_PASSWORD = os.environ.get("FRIDAY_GRAPH_PASSWORD", "")
GRAPH_SECRET = os.environ.get("FRIDAY_GRAPH_SECRET", _secrets.token_hex(32))


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

    # Three-layer memory migrations (additive on existing tables)
    for mig in [
        "ALTER TABLE world_model ADD COLUMN promoted_to TEXT DEFAULT ''",
        "ALTER TABLE world_model ADD COLUMN promoted_at TEXT",
        "ALTER TABLE world_model ADD COLUMN promoted_ref TEXT",
        "ALTER TABLE memories ADD COLUMN layer TEXT DEFAULT 'semantic'",
        "ALTER TABLE memories ADD COLUMN provenance TEXT DEFAULT '[]'",
        "ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 0.8",
        "ALTER TABLE memories ADD COLUMN last_verified TEXT",
        "ALTER TABLE memories ADD COLUMN decay_rate REAL DEFAULT 0.01",
        "ALTER TABLE entities ADD COLUMN provenance TEXT DEFAULT '[]'",
        "ALTER TABLE entities ADD COLUMN confidence REAL DEFAULT 0.8",
        "ALTER TABLE entities ADD COLUMN last_verified TEXT",
        "ALTER TABLE skills ADD COLUMN preconditions TEXT DEFAULT '[]'",
        "ALTER TABLE skills ADD COLUMN tools_needed TEXT DEFAULT '[]'",
        "ALTER TABLE skills ADD COLUMN success_count INTEGER DEFAULT 0",
        "ALTER TABLE skills ADD COLUMN failure_count INTEGER DEFAULT 0",
        "ALTER TABLE skills ADD COLUMN cost_avg REAL DEFAULT 0.0",
        "ALTER TABLE skills ADD COLUMN time_avg_sec REAL DEFAULT 0.0",
        "ALTER TABLE skills ADD COLUMN failure_domains TEXT DEFAULT '[]'",
        "ALTER TABLE skills ADD COLUMN tests TEXT DEFAULT '[]'",
        "ALTER TABLE skills ADD COLUMN maturity TEXT DEFAULT 'draft'",
        "ALTER TABLE skills ADD COLUMN examples TEXT DEFAULT '[]'",
        "ALTER TABLE insights ADD COLUMN provenance TEXT DEFAULT '[]'",
        "ALTER TABLE insights ADD COLUMN last_verified TEXT",
        "ALTER TABLE insights ADD COLUMN title TEXT DEFAULT ''",
        "ALTER TABLE insights ADD COLUMN content TEXT DEFAULT ''",
        "ALTER TABLE insights ADD COLUMN severity TEXT DEFAULT ''",
        "ALTER TABLE insights ADD COLUMN category TEXT DEFAULT ''",
        "ALTER TABLE world_model ADD COLUMN provenance TEXT DEFAULT '[]'",
        "ALTER TABLE world_model ADD COLUMN last_verified TEXT",
        # v2.13: calibration loop — store both raw user-supplied and offset-adjusted confidence
        "ALTER TABLE wm_predictions ADD COLUMN confidence_raw REAL",
        "ALTER TABLE wm_predictions ADD COLUMN confidence_adjusted REAL",
        # v2.13: proposal apply loop — track apply outcome + rollback metadata
        "ALTER TABLE proposals ADD COLUMN apply_status TEXT DEFAULT ''",
        "ALTER TABLE proposals ADD COLUMN applied_at TEXT",
        "ALTER TABLE proposals ADD COLUMN apply_error TEXT DEFAULT ''",
        "ALTER TABLE proposals ADD COLUMN backup_path TEXT DEFAULT ''",
        "ALTER TABLE proposals ADD COLUMN updated_at TEXT",
    ]:
        try:
            conn.execute(mig)
        except Exception:
            pass  # Already applied

    # v2.13: Phase 3 — performance indexes on frequently-queried foreign keys.
    # plan_tree.goal_id is filtered on goal-detail views; wm_entities.entity_key
    # is the natural lookup key + UNIQUE upsert hot path; wm_predictions.resolved
    # is the calibration loop's main filter; proposals.status drives
    # /proposal/pending and /proposal/list?status=...
    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_plan_tree_goal ON plan_tree(goal_id)",
        "CREATE INDEX IF NOT EXISTS idx_wm_entity_key ON wm_entities(entity_key)",
        "CREATE INDEX IF NOT EXISTS idx_wm_pred_resolved ON wm_predictions(resolved)",
        "CREATE INDEX IF NOT EXISTS idx_wm_pred_resolved_at ON wm_predictions(resolved_at)",
        "CREATE INDEX IF NOT EXISTS idx_proposal_status ON proposals(status)",
        "CREATE INDEX IF NOT EXISTS idx_sandbox_action ON sandbox_executions(action)",
        "CREATE INDEX IF NOT EXISTS idx_skill_maturity ON skills(maturity)",
        "CREATE INDEX IF NOT EXISTS idx_verify_subject ON verifications(subject_type, subject_id)",
    ]:
        try:
            conn.execute(idx_sql)
        except Exception:
            pass

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
        """CREATE TABLE IF NOT EXISTS active_crons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT UNIQUE,
            label TEXT,
            cron_expr TEXT,
            prompt_preview TEXT,
            registered_at TEXT,
            updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT UNIQUE,
            hypothesis TEXT,
            context TEXT,
            metric TEXT,
            min_delta REAL DEFAULT 0.05,
            min_samples INTEGER DEFAULT 10,
            variants TEXT,
            observations TEXT,
            status TEXT DEFAULT 'running',
            winner TEXT,
            conclusion TEXT,
            started_at TEXT,
            concluded_at TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            value REAL,
            unit TEXT,
            context TEXT,
            tags TEXT,
            timestamp TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_type TEXT,
            subject_id TEXT,
            check_type TEXT,
            passed INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.5,
            reason TEXT,
            evidence TEXT,
            sources TEXT,
            halluc_risk REAL DEFAULT 0.0,
            required_evidence INTEGER DEFAULT 0,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS sandbox_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id TEXT UNIQUE,
            mode TEXT,
            plan_id TEXT,
            goal_id TEXT,
            skill_id INTEGER,
            action TEXT,
            input TEXT,
            simulated_output TEXT,
            predicted_cost REAL,
            predicted_time_sec REAL,
            verdict TEXT,
            promoted_to_live INTEGER DEFAULT 0,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS plan_tree (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT,
            node_id TEXT UNIQUE,
            parent_node TEXT,
            goal_id TEXT,
            node_type TEXT,
            title TEXT,
            description TEXT,
            tool TEXT,
            expected_result TEXT,
            exit_condition TEXT,
            rollback TEXT,
            status TEXT DEFAULT 'pending',
            depth INTEGER DEFAULT 0,
            order_idx INTEGER DEFAULT 0,
            result TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS wm_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_key TEXT UNIQUE,
            name TEXT,
            type TEXT,
            state TEXT,
            attributes TEXT,
            confidence REAL DEFAULT 0.7,
            last_verified TEXT,
            created_at TEXT,
            updated_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS wm_relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            confidence REAL DEFAULT 0.6,
            evidence TEXT,
            provenance TEXT,
            last_verified TEXT,
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(subject, predicate, object)
        )""",
        """CREATE TABLE IF NOT EXISTS wm_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            actor TEXT,
            target TEXT,
            payload TEXT,
            causes TEXT,
            effects TEXT,
            occurred_at TEXT,
            created_at TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS wm_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis TEXT,
            condition TEXT,
            predicted_outcome TEXT,
            counterfactual TEXT,
            confidence REAL DEFAULT 0.5,
            due_at TEXT,
            resolved INTEGER DEFAULT 0,
            actual_outcome TEXT,
            resolved_at TEXT,
            calibration REAL,
            created_at TEXT,
            updated_at TEXT
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
app.secret_key = GRAPH_SECRET
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def _graph_auth_ok():
    return bool(GRAPH_PASSWORD) and session.get("graph_auth") is True


LOGIN_PAGE = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Memory Graph · Login</title>
<style>
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0a0f;color:#e2e8f0;font-family:-apple-system,BlinkMacSystemFont,'Inter',sans-serif}
.box{background:#0f0f18;border:1px solid #1f1f2a;padding:32px 28px;border-radius:14px;width:340px;box-shadow:0 20px 60px rgba(0,0,0,.5)}
h1{margin:0 0 4px;font-size:20px;font-weight:700}
p.sub{margin:0 0 20px;color:#64748b;font-size:13px}
label{display:block;font-size:11px;letter-spacing:.1em;color:#94a3b8;text-transform:uppercase;margin:10px 0 6px}
input{width:100%;padding:11px 12px;background:#000;border:1px solid #2a2a3a;border-radius:8px;color:#e2e8f0;font:14px/1.4 inherit;outline:none}
input:focus{border-color:#00ff41;box-shadow:0 0 0 3px rgba(0,255,65,.1)}
button{margin-top:18px;width:100%;padding:12px;background:#00ff41;color:#000;border:none;border-radius:8px;font-weight:700;letter-spacing:.05em;cursor:pointer}
button:hover{background:#fff}
.err{margin-top:12px;padding:8px 10px;background:rgba(255,45,45,.1);border:1px solid rgba(255,45,45,.3);color:#ff8a80;font-size:12px;border-radius:6px;display:__ERR_DISPLAY__}
.tag{font-family:ui-monospace,monospace;font-size:10px;color:#64748b;letter-spacing:.15em;margin-top:14px;text-align:center}
</style></head><body>
<form class="box" method="post" action="/login">
<h1>Memory Graph</h1>
<p class="sub">Acceso restringido. Ingresá credenciales.</p>
<label for="u">Usuario</label>
<input id="u" name="username" autocomplete="username" required autofocus>
<label for="p">Contraseña</label>
<input id="p" name="password" type="password" autocomplete="current-password" required>
<button type="submit">INGRESAR</button>
<div class="err">Credenciales inválidas.</div>
<div class="tag">v__VERSION__ · Friday Memory Graph</div>
</form></body></html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    if not GRAPH_PASSWORD:
        return jsonify({"error": "login disabled: FRIDAY_GRAPH_PASSWORD not configured"}), 503
    if request.method == "POST":
        u = (request.form.get("username") or "").strip()
        p = request.form.get("password") or ""
        if u == GRAPH_USER and _secrets.compare_digest(p, GRAPH_PASSWORD):
            session.permanent = True
            session["graph_auth"] = True
            nxt = request.args.get("next", "/graph")
            if not nxt.startswith("/"):
                nxt = "/graph"
            return redirect(nxt)
        page = LOGIN_PAGE.replace("__ERR_DISPLAY__", "block").replace("__VERSION__", VERSION)
        resp = make_response(page, 401)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    if _graph_auth_ok():
        return redirect("/graph")
    page = LOGIN_PAGE.replace("__ERR_DISPLAY__", "none").replace("__VERSION__", VERSION)
    resp = make_response(page)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.pop("graph_auth", None)
    return redirect("/login")


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
    limit = safe_int(request.args.get("limit", "50"), default=50, max_val=100000)
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
    """IDENTITY layer — register who/what something IS.
    Use this for stable facts: Bruno is a person, sam-assistant@agentmail.to is his
    outbound email, Claude Opus 4.6 is the model. Long-lived, rarely mutates.
    For the STATE of the world (Bruno_mood, deploy_status), use /wm/entity instead."""
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


# ── Backup / Restore ──────────────────────────────────────────────────────
# Whole-DB snapshot for disaster recovery. If the mini-pc dies, export +
# save the .db file somewhere safe; later POST it back to /backup/import on
# a fresh install and restart the server to resume where things stopped.

@app.route("/backup/info")
def backup_info():
    import shutil
    db_path = Path(DB_PATH)
    size = db_path.stat().st_size if db_path.exists() else 0
    mtime = datetime.datetime.fromtimestamp(db_path.stat().st_mtime).isoformat() if db_path.exists() else None
    # List pre-import backups
    parent = db_path.parent
    backups = []
    for f in sorted(parent.glob(f"{db_path.name}.pre-import-*"), reverse=True):
        backups.append({
            "name": f.name,
            "size_bytes": f.stat().st_size,
            "created": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    # Disk space
    disk = shutil.disk_usage(str(parent))
    return jsonify({
        "db_path": str(db_path),
        "db_size_bytes": size,
        "db_size_mb": round(size / (1024*1024), 2),
        "db_last_modified": mtime,
        "version": VERSION,
        "disk_free_mb": round(disk.free / (1024*1024), 2),
        "pre_import_backups": backups,
    })


@app.route("/backup/export")
def backup_export():
    """Generate a consistent snapshot of the DB via VACUUM INTO and stream it
    as a download. Safe to call while the server is serving requests.
    Optional ?format=dump returns a SQL dump (.sql) instead of the binary .db."""
    import tempfile
    fmt = request.args.get("format", "db")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if fmt == "dump":
        # Plain-SQL dump (portable, diffable, slower)
        src = sqlite3.connect(DB_PATH)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False, encoding="utf-8")
        for line in src.iterdump():
            tmp.write(f"{line}\n")
        tmp.close()
        src.close()
        return send_file(
            tmp.name,
            as_attachment=True,
            download_name=f"friday-memory-{ts}.sql",
            mimetype="application/sql",
        )
    # Default: .db binary snapshot via VACUUM INTO (consistent even with writers)
    snap_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    os.remove(snap_path)  # VACUUM INTO requires destination to not exist
    src = sqlite3.connect(DB_PATH)
    try:
        src.execute("VACUUM INTO ?", [snap_path])
    finally:
        src.close()
    return send_file(
        snap_path,
        as_attachment=True,
        download_name=f"friday-memory-{ts}.db",
        mimetype="application/x-sqlite3",
    )


@app.route("/backup/import", methods=["POST"])
def backup_import():
    """Replace the live DB with an uploaded snapshot. The previous DB is kept
    as <db>.pre-import-<ts>. After import, RESTART the server to reload
    connections — the API will keep answering on the old DB until restart."""
    import shutil
    import tempfile
    if "file" not in request.files:
        return jsonify({"error": "upload the snapshot with form field 'file'"}), 400
    up = request.files["file"]
    if not up.filename:
        return jsonify({"error": "empty filename"}), 400
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp.name
    tmp.close()
    up.save(tmp_path)
    # Validate it's a real SQLite with at least one expected table
    try:
        conn = sqlite3.connect(tmp_path)
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        conn.close()
        tables = [r[0] for r in rows]
        if not tables:
            os.remove(tmp_path)
            return jsonify({"error": "uploaded file has no tables"}), 400
        expected = {"conversations", "memories", "entities"}
        missing = expected - set(tables)
        if missing:
            os.remove(tmp_path)
            return jsonify({"error": f"uploaded file is missing core tables: {sorted(missing)}"}), 400
    except sqlite3.DatabaseError as e:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return jsonify({"error": f"not a valid SQLite database: {e}"}), 400
    # Backup current DB
    db_path = Path(DB_PATH)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.parent / f"{db_path.name}.pre-import-{ts}"
    shutil.copy2(DB_PATH, str(backup_path))
    # Swap in uploaded file
    shutil.copy2(tmp_path, DB_PATH)
    os.remove(tmp_path)
    new_size = Path(DB_PATH).stat().st_size
    return jsonify({
        "status": "imported",
        "new_db_size_bytes": new_size,
        "new_db_size_mb": round(new_size / (1024*1024), 2),
        "tables_found": sorted(tables),
        "previous_db_backup": str(backup_path),
        "restart_required": True,
        "restart_note": "Restart the server (kill + nohup) to reload sqlite_utils connections with the imported DB. Until restart the API keeps serving the old DB from memory.",
    })


@app.route("/graph")
def graph():
    if GRAPH_PASSWORD and not _graph_auth_ok():
        return redirect("/login?next=/graph")
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
        "SELECT id, name, trigger_pattern, description, steps, times_used, "
        "last_used, created_at, success_count, failure_count, maturity, "
        "cost_avg, time_avg_sec FROM skills ORDER BY times_used DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "name": r[1], "trigger_pattern": r[2], "description": r[3],
                "steps": r[4], "times_used": r[5], "last_used": r[6], "created_at": r[7],
                "success_count": r[8] or 0, "failure_count": r[9] or 0,
                "maturity": r[10] or "draft",
                "cost_avg": r[11] or 0.0, "time_avg_sec": r[12] or 0.0} for r in rows]
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
        "title": data.get("title", ""),
        "content": data.get("content", ""),
        "severity": data.get("severity", ""),
        "category": data.get("category", ""),
        "created_at": ts,
    }, pk="id")
    last_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"status": "ok", "id": last_id})


@app.route("/insight/active")
def insight_active():
    db = get_db()
    ts = now_iso()
    rows = db.execute(
        "SELECT id, type, pattern, evidence, confidence, valid_until, created_at, title, content, severity, category FROM insights WHERE valid_until IS NULL OR valid_until > ? ORDER BY confidence DESC",
        [ts]
    ).fetchall()
    results = [{"id": r[0], "type": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "valid_until": r[5], "created_at": r[6],
                "title": r[7], "content": r[8], "severity": r[9], "category": r[10]}
               for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/insight/list")
def insight_list():
    db = get_db()
    limit = safe_int(request.args.get("limit", "100"), default=100)
    rows = db.execute(
        "SELECT id, type, pattern, evidence, confidence, valid_until, created_at, title, content, severity, category FROM insights ORDER BY created_at DESC LIMIT ?",
        [limit]
    ).fetchall()
    results = [{"id": r[0], "type": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "valid_until": r[5], "created_at": r[6],
                "title": r[7], "content": r[8], "severity": r[9], "category": r[10]}
               for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/insight/<int:id>", methods=["GET"])
def insight_get(id):
    db = get_db()
    rows = db.execute(
        "SELECT id, type, pattern, evidence, confidence, valid_until, created_at, title, content, severity, category FROM insights WHERE id = ?",
        [id]
    ).fetchall()
    if not rows:
        return jsonify({"error": f"Insight {id} not found"}), 404
    r = rows[0]
    return jsonify({"id": r[0], "type": r[1], "pattern": r[2], "evidence": r[3],
                    "confidence": r[4], "valid_until": r[5], "created_at": r[6],
                    "title": r[7], "content": r[8], "severity": r[9], "category": r[10]})


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


@app.route("/proposal/list")
def proposal_list():
    db = get_db()
    status = request.args.get("status")
    if status:
        rows = db.execute(
            "SELECT id, file_path, change_type, description, diff_preview, status, created_at, resolved_at FROM proposals WHERE status = ? ORDER BY created_at DESC",
            [status]
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, file_path, change_type, description, diff_preview, status, created_at, resolved_at FROM proposals ORDER BY created_at DESC"
        ).fetchall()
    results = [{"id": r[0], "file_path": r[1], "change_type": r[2], "description": r[3],
                "diff_preview": r[4], "status": r[5], "created_at": r[6], "resolved_at": r[7]} for r in rows]
    return jsonify({"count": len(results), "results": results})


# --- v2.13: proposal apply loop ---

PROPOSAL_BACKUP_DIR = Path.home() / ".friday" / "proposal_backups"
PROPOSAL_ALLOWED_ROOTS = [Path.home() / "proyectos", Path.home() / ".claude"]


def _is_allowed_proposal_path(file_path: str) -> bool:
    if not file_path:
        return False
    try:
        p = Path(file_path).resolve()
    except Exception:
        return False
    for root in PROPOSAL_ALLOWED_ROOTS:
        try:
            p.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def _is_unified_diff(text: str) -> bool:
    if not text:
        return False
    head = text.lstrip()
    return (head.startswith("--- ") or head.startswith("diff --git")) and "@@" in text


def _apply_proposal_diff(proposal_id: int, file_path: str, diff_text: str):
    """Apply a unified diff to file_path with backup + auto-rollback on failure.
    Returns (apply_status, error_msg, backup_path).
    apply_status ∈ {applied, apply_failed, skipped_invalid_diff, skipped_unsafe_path,
                    skipped_no_target}."""
    if not _is_allowed_proposal_path(file_path):
        return ("skipped_unsafe_path",
                f"path not under allowed roots ({[str(r) for r in PROPOSAL_ALLOWED_ROOTS]})",
                None)
    if not _is_unified_diff(diff_text):
        return ("skipped_invalid_diff",
                "diff_preview is not a unified diff (--- / @@)",
                None)
    target = Path(file_path)
    if not target.exists():
        return ("skipped_no_target", f"target file does not exist: {file_path}", None)
    try:
        PROPOSAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return ("apply_failed", f"backup dir create failed: {e}", None)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = PROPOSAL_BACKUP_DIR / f"{proposal_id}_{ts}_{target.name}"
    try:
        shutil.copy2(target, backup)
    except Exception as e:
        return ("apply_failed", f"backup failed: {e}", None)
    try:
        result = subprocess.run(
            ["patch", "-p0", "--no-backup-if-mismatch", str(target)],
            input=diff_text, text=True, capture_output=True, timeout=10
        )
        if result.returncode != 0:
            try:
                shutil.copy2(backup, target)
            except Exception:
                pass
            err = (result.stderr or result.stdout or "")[:300]
            return ("apply_failed", f"patch exit {result.returncode}: {err}", str(backup))
        return ("applied", None, str(backup))
    except subprocess.TimeoutExpired:
        try:
            shutil.copy2(backup, target)
        except Exception:
            pass
        return ("apply_failed", "patch timeout 10s (rolled back)", str(backup))
    except FileNotFoundError:
        return ("apply_failed", "`patch` binary not found in PATH", str(backup))
    except Exception as e:
        try:
            shutil.copy2(backup, target)
        except Exception:
            pass
        return ("apply_failed", str(e)[:300], str(backup))


@app.route("/proposal/<int:proposal_id>/approve", methods=["PUT", "POST", "PATCH"])
def proposal_approve(proposal_id):
    """Approve a proposal AND attempt to auto-apply its diff (v2.13 closed loop).
    Body: {dry_run: false} — set dry_run=true to mark approved without applying."""
    db = get_db()
    row = db.execute(
        "SELECT file_path, diff_preview FROM proposals WHERE id = ?",
        [proposal_id]).fetchone()
    if not row:
        return jsonify({"error": "Proposal not found"}), 404
    data = request.json or {}
    dry_run = bool(data.get("dry_run", False))
    file_path, diff_text = row[0], row[1]
    ts = now_iso()
    if dry_run:
        db.execute(
            "UPDATE proposals SET status = 'approved', resolved_at = ?, "
            "apply_status = 'dry_run', updated_at = ? WHERE id = ?",
            [ts, ts, proposal_id])
        return jsonify({"status": "approved", "id": proposal_id, "apply_status": "dry_run"})
    apply_status, apply_error, backup_path = _apply_proposal_diff(
        proposal_id, file_path, diff_text)
    db.execute(
        "UPDATE proposals SET status = 'approved', resolved_at = ?, "
        "apply_status = ?, applied_at = ?, apply_error = ?, "
        "backup_path = ?, updated_at = ? WHERE id = ?",
        [ts, apply_status,
         ts if apply_status == "applied" else None,
         apply_error or "",
         backup_path or "",
         ts, proposal_id])
    return jsonify({
        "status": "approved",
        "id": proposal_id,
        "apply_status": apply_status,
        "apply_error": apply_error,
        "backup_path": backup_path,
    })


@app.route("/proposal/<int:proposal_id>/reject", methods=["PUT", "POST", "PATCH"])
def proposal_reject(proposal_id):
    db = get_db()
    ts = now_iso()
    db.execute(
        "UPDATE proposals SET status = 'rejected', resolved_at = ?, updated_at = ? WHERE id = ?",
        [ts, ts, proposal_id])
    return jsonify({"status": "rejected", "id": proposal_id})


@app.route("/proposal/<int:proposal_id>/rollback", methods=["POST"])
def proposal_rollback(proposal_id):
    """Restore the file from the backup created when the proposal was applied."""
    db = get_db()
    row = db.execute(
        "SELECT file_path, backup_path, apply_status FROM proposals WHERE id = ?",
        [proposal_id]).fetchone()
    if not row:
        return jsonify({"error": "not found"}), 404
    file_path, backup_path, apply_status = row[0], row[1], row[2]
    if apply_status != "applied":
        return jsonify({"error": f"cannot rollback (apply_status={apply_status})"}), 400
    if not backup_path or not Path(backup_path).exists():
        return jsonify({"error": "backup missing"}), 404
    try:
        shutil.copy2(backup_path, file_path)
    except Exception as e:
        return jsonify({"error": f"rollback failed: {e}"}), 500
    ts = now_iso()
    db.execute(
        "UPDATE proposals SET apply_status = 'rolled_back', updated_at = ? WHERE id = ?",
        [ts, proposal_id])
    return jsonify({"status": "rolled_back", "id": proposal_id, "restored_from": backup_path})


# -- World Model --

@app.route("/worldmodel", methods=["POST"])
def worldmodel_create():
    """SOFT OBSERVATION INBOX — loose pattern that hasn't earned structure yet.
    Reflection/preference crons dump observations here first. Once a pattern becomes
    testable (prediction), causal (event with causes/effects), or
    subject-predicate-object (relation), promote it with POST /worldmodel/<id>/promote.

    If a matching pattern+category already exists, occurrences+confidence are bumped
    instead of creating a duplicate."""
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
        "SELECT id, category, pattern, evidence, confidence, occurrences, first_seen, last_seen, expires_at, created_at, promoted_to, promoted_ref FROM world_model WHERE confidence >= 0.4 AND (expires_at = '' OR expires_at IS NULL OR expires_at > ?) AND (promoted_to = '' OR promoted_to IS NULL) ORDER BY confidence DESC, occurrences DESC",
        [ts]
    ).fetchall()
    results = [{"id": r[0], "category": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "occurrences": r[5], "first_seen": r[6],
                "last_seen": r[7], "expires_at": r[8], "created_at": r[9],
                "promoted_to": r[10] or "", "promoted_ref": r[11] or ""} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/worldmodel/list")
def worldmodel_list():
    """List all world model entries."""
    db = get_db()
    category = request.args.get("category", "")
    limit = safe_int(request.args.get("limit", "100"), default=100)
    if category:
        rows = db.execute(
            "SELECT id, category, pattern, evidence, confidence, occurrences, first_seen, last_seen, expires_at, created_at, promoted_to, promoted_ref FROM world_model WHERE category = ? ORDER BY confidence DESC LIMIT ?",
            [category, limit]
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, category, pattern, evidence, confidence, occurrences, first_seen, last_seen, expires_at, created_at, promoted_to, promoted_ref FROM world_model ORDER BY category, confidence DESC LIMIT ?",
            [limit]
        ).fetchall()
    results = [{"id": r[0], "category": r[1], "pattern": r[2], "evidence": r[3],
                "confidence": r[4], "occurrences": r[5], "first_seen": r[6],
                "last_seen": r[7], "expires_at": r[8], "created_at": r[9],
                "promoted_to": r[10] or "", "promoted_ref": r[11] or ""} for r in rows]
    return jsonify({"count": len(results), "results": results})


@app.route("/worldmodel/<int:entry_id>", methods=["DELETE"])
def worldmodel_delete(entry_id):
    db = get_db()
    existing = db.execute("SELECT id FROM world_model WHERE id = ?", [entry_id]).fetchone()
    if not existing:
        return jsonify({"error": f"World model entry {entry_id} not found"}), 404
    db.execute("DELETE FROM world_model WHERE id = ?", [entry_id])
    return jsonify({"status": "deleted", "id": entry_id})


@app.route("/worldmodel/<int:entry_id>/promote", methods=["POST"])
def worldmodel_promote(entry_id):
    """Graduate a soft observation to structured knowledge.
    Body: {target: 'event' | 'relation' | 'prediction', ...fields specific to target}

    The source row is kept (its history matters) but marked promoted_to=<target>
    and promoted_ref=<new id/key> so audits can follow the chain."""
    db = get_db()
    row = db.execute(
        "SELECT id, category, pattern, evidence, confidence, occurrences, promoted_to FROM world_model WHERE id = ?",
        [entry_id]).fetchone()
    if not row:
        return jsonify({"error": f"World model entry {entry_id} not found"}), 404
    if row[6]:
        return jsonify({"error": f"Already promoted to {row[6]}"}), 400
    data = request.json or {}
    target = (data.get("target") or "").lower().strip()
    if target not in {"event", "relation", "prediction"}:
        return jsonify({"error": "target must be 'event' | 'relation' | 'prediction'"}), 400

    ts = now_iso()
    provenance_entry = f"worldmodel:{entry_id}"
    promoted_ref = ""

    def _s(v):
        return v if isinstance(v, str) else json.dumps(v or [], ensure_ascii=False)

    if target == "event":
        etype = (data.get("event_type") or row[1] or "observed").strip()
        db["wm_events"].insert({
            "event_type": etype,
            "actor": data.get("actor", ""),
            "target": data.get("target_actor", ""),
            "payload": _s(data.get("payload", {"pattern": row[2], "evidence": row[3]})),
            "causes": _s(data.get("causes", [])),
            "effects": _s(data.get("effects", [])),
            "occurred_at": data.get("occurred_at", ts),
            "created_at": ts,
        })
        promoted_ref = f"event:{etype}"
    elif target == "relation":
        subject = (data.get("subject") or "").strip()
        predicate = (data.get("predicate") or row[1] or "").strip()
        obj = (data.get("object") or "").strip()
        if not (subject and predicate and obj):
            return jsonify({"error": "relation needs subject, predicate, object"}), 400
        db["wm_relations"].insert({
            "subject": subject, "predicate": predicate, "object": obj,
            "confidence": clamp_float(data.get("confidence", row[4] or 0.6), default=0.6),
            "evidence": row[3] or "",
            "provenance": json.dumps([provenance_entry], ensure_ascii=False),
            "last_verified": ts, "created_at": ts, "updated_at": ts,
        })
        promoted_ref = f"{subject}→{predicate}→{obj}"
    else:  # prediction
        hypothesis = (data.get("hypothesis") or row[2] or "").strip()
        outcome = (data.get("predicted_outcome") or "").strip()
        if not hypothesis or not outcome:
            return jsonify({"error": "prediction needs hypothesis and predicted_outcome"}), 400
        db["wm_predictions"].insert({
            "hypothesis": hypothesis,
            "condition": data.get("condition", ""),
            "predicted_outcome": outcome,
            "counterfactual": data.get("counterfactual", ""),
            "confidence": clamp_float(data.get("confidence", row[4] or 0.5), default=0.5),
            "due_at": data.get("due_at", ""),
            "resolved": 0,
            "actual_outcome": "",
            "resolved_at": "",
            "calibration": None,
            "created_at": ts,
            "updated_at": ts,
        })
        promoted_ref = f"prediction:{hypothesis[:40]}"

    db.execute(
        "UPDATE world_model SET promoted_to = ?, promoted_at = ?, promoted_ref = ?, updated_at = ? WHERE id = ?",
        [target, ts, promoted_ref, ts, entry_id])

    return jsonify({
        "status": "promoted",
        "source_id": entry_id,
        "target": target,
        "promoted_ref": promoted_ref,
    })


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


# -- Three-layer memory --

@app.route("/memory/episodic", methods=["GET"])
def memory_episodic():
    """Episodic memory: what happened, when, with what outcome.
    Pulls from conversations + world_model events."""
    db = get_db()
    hours = safe_int(request.args.get("hours"), default=72, max_val=8760)
    limit = safe_int(request.args.get("limit"), default=100, max_val=1000)
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).isoformat()
    rows = db.execute(
        "SELECT id, timestamp, role, channel, content, importance FROM conversations WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
        [cutoff, limit]).fetchall()
    events = [{"id": r[0], "timestamp": r[1], "role": r[2], "channel": r[3],
               "content": r[4][:500], "importance": r[5], "layer": "episodic", "source": "conversation"} for r in rows]
    wm_rows = db.execute(
        "SELECT id, category, pattern, first_seen, last_seen, confidence FROM world_model WHERE last_seen >= ? ORDER BY last_seen DESC LIMIT ?",
        [cutoff, limit]).fetchall()
    for r in wm_rows:
        events.append({"id": r[0], "timestamp": r[4], "category": r[1], "pattern": r[2],
                       "first_seen": r[3], "confidence": r[5], "layer": "episodic", "source": "world_model"})
    events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify({"count": len(events), "hours": hours, "events": events[:limit]})


@app.route("/memory/semantic", methods=["GET"])
def memory_semantic():
    """Semantic memory: stable facts, concepts, relations.
    Combines memories (layer=semantic) + entities."""
    db = get_db()
    mtype = request.args.get("type", "")
    limit = safe_int(request.args.get("limit"), default=100, max_val=1000)
    if mtype:
        mem_rows = db.execute(
            "SELECT id, type, name, description, content, confidence, provenance, last_verified, updated_at FROM memories WHERE type = ? AND (layer = 'semantic' OR layer IS NULL) ORDER BY updated_at DESC LIMIT ?",
            [mtype, limit]).fetchall()
    else:
        mem_rows = db.execute(
            "SELECT id, type, name, description, content, confidence, provenance, last_verified, updated_at FROM memories WHERE (layer = 'semantic' OR layer IS NULL) ORDER BY updated_at DESC LIMIT ?",
            [limit]).fetchall()
    memories = [{"id": r[0], "type": r[1], "name": r[2], "description": r[3],
                 "content": r[4], "confidence": r[5] or 0.8,
                 "provenance": _safe_json(r[6], []),
                 "last_verified": r[7], "updated_at": r[8], "layer": "semantic", "source": "memory"} for r in mem_rows]
    ent_rows = db.execute(
        "SELECT id, name, type, details, confidence, provenance, last_verified, updated_at FROM entities ORDER BY updated_at DESC LIMIT ?",
        [limit]).fetchall()
    entities = [{"id": r[0], "name": r[1], "type": r[2], "details": r[3],
                 "confidence": r[4] or 0.8,
                 "provenance": _safe_json(r[5], []),
                 "last_verified": r[6], "updated_at": r[7], "layer": "semantic", "source": "entity"} for r in ent_rows]
    return jsonify({"count": len(memories) + len(entities), "memories": memories, "entities": entities})


@app.route("/memory/procedural", methods=["GET"])
def memory_procedural():
    """Procedural memory: skills, playbooks, heuristics."""
    db = get_db()
    limit = safe_int(request.args.get("limit"), default=100, max_val=500)
    rows = db.execute(
        "SELECT id, name, trigger_pattern, description, steps, preconditions, tools_needed, success_count, failure_count, cost_avg, time_avg_sec, failure_domains, maturity, times_used, last_used, created_at FROM skills ORDER BY times_used DESC, created_at DESC LIMIT ?",
        [limit]).fetchall()
    skills = []
    for r in rows:
        total = (r[7] or 0) + (r[8] or 0)
        sr = (r[7] / total) if total > 0 else None
        skills.append({
            "id": r[0], "name": r[1], "trigger_pattern": r[2], "description": r[3],
            "steps": _safe_json(r[4], []), "preconditions": _safe_json(r[5], []),
            "tools_needed": _safe_json(r[6], []),
            "success_count": r[7] or 0, "failure_count": r[8] or 0,
            "success_rate": sr,
            "cost_avg": r[9] or 0.0, "time_avg_sec": r[10] or 0.0,
            "failure_domains": _safe_json(r[11], []),
            "maturity": r[12] or "draft",
            "times_used": r[13] or 0, "last_used": r[14], "created_at": r[15],
            "layer": "procedural", "source": "skill",
        })
    return jsonify({"count": len(skills), "skills": skills})


def _safe_json(s, default):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


@app.route("/memory/<int:mem_id>/verify", methods=["POST"])
def memory_verify(mem_id):
    """Mark a memory as re-verified (resets decay). Body optional: {boost: 0.1}"""
    db = get_db()
    row = db.execute("SELECT id, confidence FROM memories WHERE id = ?", [mem_id]).fetchone()
    if not row:
        return jsonify({"error": f"Memory {mem_id} not found"}), 404
    data = request.json or {}
    boost = clamp_float(data.get("boost", 0.1), default=0.1, lo=0.0, hi=0.5)
    new_conf = min(1.0, (row[1] or 0.8) + boost)
    ts = now_iso()
    db.execute("UPDATE memories SET confidence = ?, last_verified = ?, updated_at = ? WHERE id = ?",
               [new_conf, ts, ts, mem_id])
    return jsonify({"status": "verified", "id": mem_id, "confidence": new_conf, "last_verified": ts})


@app.route("/memory/decay", methods=["POST"])
def memory_decay():
    """Apply confidence decay across memories, entities, world_model based on age.
    Body optional: {halflife_days: 60}"""
    db = get_db()
    data = request.json or {}
    halflife_days = float(data.get("halflife_days", 60.0) or 60.0)
    now = datetime.datetime.now(datetime.timezone.utc)
    decayed = {"memories": 0, "entities": 0, "world_model": 0}

    # v2.13 Phase 3: hardened against accidental SQL injection via column-name
    # f-string. Only allowlisted (table, cols) tuples are accepted. Caller
    # must pass exact identifiers from this set.
    _DECAY_ALLOWLIST = {
        ("memories", "id", "confidence", "last_verified", "updated_at"),
        ("entities", "id", "confidence", "last_verified", "updated_at"),
        ("world_model", "id", "confidence", "last_verified", "last_seen"),
    }

    def _decay_table(table, id_col, conf_col, verified_col, fallback_col):
        if (table, id_col, conf_col, verified_col, fallback_col) not in _DECAY_ALLOWLIST:
            raise ValueError(f"unauthorized decay target: {(table, id_col, conf_col, verified_col, fallback_col)}")
        rows = db.execute(
            f"SELECT {id_col}, {conf_col}, {verified_col}, {fallback_col} FROM {table}"
        ).fetchall()
        changed = 0
        for r in rows:
            ref_ts = r[2] or r[3]
            if not ref_ts:
                continue
            try:
                ts = datetime.datetime.fromisoformat(ref_ts.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                age_days = (now - ts).total_seconds() / 86400.0
                if age_days <= halflife_days:
                    continue
                old_conf = r[1] or 0.8
                factor = 0.5 ** ((age_days - halflife_days) / halflife_days)
                new_conf = max(0.05, old_conf * factor)
                if abs(new_conf - old_conf) >= 0.01:
                    db.execute(f"UPDATE {table} SET {conf_col} = ? WHERE {id_col} = ?",
                               [new_conf, r[0]])
                    changed += 1
            except Exception:
                continue
        return changed

    decayed["memories"] = _decay_table("memories", "id", "confidence", "last_verified", "updated_at")
    decayed["entities"] = _decay_table("entities", "id", "confidence", "last_verified", "updated_at")
    decayed["world_model"] = _decay_table("world_model", "id", "confidence", "last_verified", "last_seen")

    # v2.13: insights have a valid_until field — actually enforce expiry.
    # Mark expired insights as confidence=0 + add 'expired' tag in category if empty.
    expired_insights = 0
    now_iso_str = now.isoformat()
    rows = db.execute(
        "SELECT id, valid_until, confidence, category FROM insights "
        "WHERE valid_until IS NOT NULL AND valid_until != '' AND valid_until < ?",
        [now_iso_str]).fetchall()
    for r in rows:
        iid, _, conf, cat = r[0], r[1], r[2], r[3] or ""
        if (conf or 0.0) > 0.05 or "expired" not in cat:
            new_cat = cat if "expired" in cat else (f"{cat},expired" if cat else "expired")
            db.execute(
                "UPDATE insights SET confidence = 0.0, category = ? WHERE id = ?",
                [new_cat, iid])
            expired_insights += 1
    decayed["insights_expired"] = expired_insights

    return jsonify({"status": "decayed", "halflife_days": halflife_days, "changes": decayed})


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


@app.route("/harness/activity", methods=["GET"])
def harness_activity():
    """Write activity per harness table. Used by Brain dashboard to flag stale sections.
    Query: ?hours=24 (default), or ?days=7"""
    db = get_db()
    hours = safe_int(request.args.get("hours"), default=24, max_val=8760)
    if request.args.get("days"):
        hours = safe_int(request.args.get("days"), default=1, max_val=365) * 24
    since = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).isoformat()

    tracked = {
        "goals": "created_at",
        "plan_tree": "created_at",
        "capabilities": "updated_at",
        "experiments": "created_at",
        "proposals": "created_at",
        "sandbox_executions": "created_at",
        "verifications": "created_at",
        "wm_entities": "updated_at",
        "wm_events": "occurred_at",
        "wm_predictions": "created_at",
        "wm_relations": "created_at",
        "world_model": "updated_at",
        "entities": "updated_at",
        "memories": "updated_at",
        "insights": "created_at",
        "preferences": "created_at",
        "reflections": "created_at",
        "skills": "created_at",
        "metrics": "timestamp",
    }
    activity = {}
    for table, ts_col in tracked.items():
        try:
            recent = db.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {ts_col} >= ?", [since]).fetchone()[0]
            total = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            last = db.execute(f"SELECT MAX({ts_col}) FROM {table}").fetchone()[0]
            activity[table] = {"recent": recent, "total": total, "last": last}
        except Exception as e:
            activity[table] = {"error": str(e)}
    return jsonify({"since": since, "hours": hours, "activity": activity})


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


# -- Hierarchical Planner --

def _plan_node_row_to_dict(r):
    return {
        "id": r[0], "plan_id": r[1], "node_id": r[2], "parent_node": r[3],
        "goal_id": r[4], "node_type": r[5], "title": r[6], "description": r[7],
        "tool": r[8], "expected_result": r[9], "exit_condition": r[10],
        "rollback": r[11], "status": r[12], "depth": r[13], "order_idx": r[14],
        "result": r[15], "created_at": r[16], "updated_at": r[17],
    }


_PLAN_NODE_COLS = "id, plan_id, node_id, parent_node, goal_id, node_type, title, description, tool, expected_result, exit_condition, rollback, status, depth, order_idx, result, created_at, updated_at"


@app.route("/plan", methods=["POST"])
def plan_create():
    """Create a new plan root node. Body: {goal_id?, title, description?, tool?, expected_result?, exit_condition?, rollback?}"""
    db = get_db()
    data = request.json or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title required"}), 400
    ts = now_iso()
    plan_id = data.get("plan_id") or f"p_{uuid.uuid4().hex[:8]}"
    node_id = f"{plan_id}_root"
    row = {
        "plan_id": plan_id,
        "node_id": node_id,
        "parent_node": "",
        "goal_id": data.get("goal_id", ""),
        "node_type": "goal",
        "title": title,
        "description": data.get("description", ""),
        "tool": data.get("tool", ""),
        "expected_result": data.get("expected_result", ""),
        "exit_condition": data.get("exit_condition", ""),
        "rollback": data.get("rollback", ""),
        "status": "pending",
        "depth": 0,
        "order_idx": 0,
        "result": "",
        "created_at": ts,
        "updated_at": ts,
    }
    db["plan_tree"].insert(row)
    created = db.execute(f"SELECT {_PLAN_NODE_COLS} FROM plan_tree WHERE node_id = ?", [node_id]).fetchone()
    return jsonify({"status": "created", "plan_id": plan_id, "root": _plan_node_row_to_dict(created)})


@app.route("/plan/<plan_id>", methods=["GET"])
def plan_get(plan_id):
    db = get_db()
    rows = db.execute(
        f"SELECT {_PLAN_NODE_COLS} FROM plan_tree WHERE plan_id = ? ORDER BY depth, order_idx",
        [plan_id]).fetchall()
    if not rows:
        return jsonify({"error": f"Plan {plan_id} not found"}), 404
    nodes = [_plan_node_row_to_dict(r) for r in rows]
    return jsonify({"plan_id": plan_id, "count": len(nodes), "nodes": nodes})


@app.route("/plan/list", methods=["GET"])
def plan_list():
    db = get_db()
    goal_id = request.args.get("goal_id", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    if goal_id:
        rows = db.execute(
            "SELECT DISTINCT plan_id, MAX(created_at) as cat FROM plan_tree WHERE goal_id = ? GROUP BY plan_id ORDER BY cat DESC LIMIT ?",
            [goal_id, limit]).fetchall()
    else:
        rows = db.execute(
            "SELECT DISTINCT plan_id, MAX(created_at) as cat FROM plan_tree GROUP BY plan_id ORDER BY cat DESC LIMIT ?",
            [limit]).fetchall()
    plans = []
    for r in rows:
        pid = r[0]
        root = db.execute(
            "SELECT title, goal_id, status, depth FROM plan_tree WHERE plan_id = ? ORDER BY depth ASC LIMIT 1",
            [pid]).fetchone()
        cnt = db.execute("SELECT COUNT(*) FROM plan_tree WHERE plan_id = ?", [pid]).fetchone()[0]
        plans.append({"plan_id": pid, "title": root[0] if root else "",
                      "goal_id": root[1] if root else "",
                      "root_status": root[2] if root else "", "nodes": cnt, "created_at": r[1]})
    return jsonify({"count": len(plans), "plans": plans})


@app.route("/plan/<plan_id>/node", methods=["POST"])
def plan_add_node(plan_id):
    """Add a child node. Body: {parent_node, node_type, title, description?, tool?, expected_result?, exit_condition?, rollback?, order_idx?}"""
    db = get_db()
    data = request.json or {}
    parent_node = data.get("parent_node") or ""
    if not parent_node:
        return jsonify({"error": "parent_node required"}), 400
    parent_row = db.execute(
        "SELECT plan_id, depth, goal_id FROM plan_tree WHERE node_id = ?", [parent_node]).fetchone()
    if not parent_row or parent_row[0] != plan_id:
        return jsonify({"error": f"parent_node {parent_node} not in plan {plan_id}"}), 404
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"error": "title required"}), 400
    ts = now_iso()
    node_id = f"{plan_id}_n{uuid.uuid4().hex[:6]}"
    row = {
        "plan_id": plan_id,
        "node_id": node_id,
        "parent_node": parent_node,
        "goal_id": parent_row[2],
        "node_type": data.get("node_type", "action"),
        "title": title,
        "description": data.get("description", ""),
        "tool": data.get("tool", ""),
        "expected_result": data.get("expected_result", ""),
        "exit_condition": data.get("exit_condition", ""),
        "rollback": data.get("rollback", ""),
        "status": "pending",
        "depth": parent_row[1] + 1,
        "order_idx": safe_int(data.get("order_idx", 0), default=0, min_val=0, max_val=10000),
        "result": "",
        "created_at": ts,
        "updated_at": ts,
    }
    db["plan_tree"].insert(row)
    created = db.execute(f"SELECT {_PLAN_NODE_COLS} FROM plan_tree WHERE node_id = ?", [node_id]).fetchone()
    return jsonify({"status": "created", "node": _plan_node_row_to_dict(created)})


@app.route("/plan/node/<node_id>", methods=["PATCH"])
def plan_update_node(node_id):
    db = get_db()
    existing = db.execute("SELECT id FROM plan_tree WHERE node_id = ?", [node_id]).fetchone()
    if not existing:
        return jsonify({"error": f"Node {node_id} not found"}), 404
    data = request.json or {}
    updatable = {"title", "description", "tool", "expected_result", "exit_condition",
                 "rollback", "status", "result", "order_idx"}
    sets = []
    vals = []
    for k, v in data.items():
        if k not in updatable:
            continue
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return jsonify({"error": "no updatable fields"}), 400
    sets.append("updated_at = ?")
    vals.append(now_iso())
    vals.append(node_id)
    db.execute(f"UPDATE plan_tree SET {', '.join(sets)} WHERE node_id = ?", vals)
    row = db.execute(f"SELECT {_PLAN_NODE_COLS} FROM plan_tree WHERE node_id = ?", [node_id]).fetchone()
    return jsonify({"status": "updated", "node": _plan_node_row_to_dict(row)})


@app.route("/plan/<plan_id>", methods=["DELETE"])
def plan_delete(plan_id):
    db = get_db()
    deleted = db.execute("DELETE FROM plan_tree WHERE plan_id = ?", [plan_id]).rowcount
    return jsonify({"status": "deleted", "plan_id": plan_id, "nodes_removed": deleted})


# -- Structured World Model (entities / relations / events / predictions) --
#
# Split from /entity by design:
#   /entity       → IDENTITY (who something IS)
#   /wm/entity    → STATE (current state of the world, decays)
# Example: Bruno is stored once in /entity; "Bruno_mood=cansado" lives in /wm/entity
# and last_verified matters.

@app.route("/wm/entity", methods=["POST"])
def wm_entity_create():
    """STATE layer — current state of something in the world.
    Name is the subject (e.g. 'Bruno_mood', 'deploy_status', 'memory_api_health').
    The `state` field carries the current value ('cansado', 'pending', 'ok').
    Repeated POSTs overwrite; last_verified is the freshness clock."""
    db = get_db()
    data = request.json or {}
    name = (data.get("name") or "").strip()
    etype = (data.get("type") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    entity_key = data.get("entity_key") or f"{etype}:{name}".lower()
    ts = now_iso()
    attrs = data.get("attributes", {})
    if not isinstance(attrs, str):
        attrs = json.dumps(attrs, ensure_ascii=False)
    existing = db.execute("SELECT id FROM wm_entities WHERE entity_key = ?", [entity_key]).fetchone()
    if existing:
        db.execute(
            "UPDATE wm_entities SET name = ?, type = ?, state = ?, attributes = ?, confidence = ?, last_verified = ?, updated_at = ? WHERE entity_key = ?",
            [name, etype, data.get("state", ""), attrs,
             clamp_float(data.get("confidence", 0.7), default=0.7), ts, ts, entity_key])
        return jsonify({"status": "updated", "entity_key": entity_key})
    db["wm_entities"].insert({
        "entity_key": entity_key, "name": name, "type": etype,
        "state": data.get("state", ""), "attributes": attrs,
        "confidence": clamp_float(data.get("confidence", 0.7), default=0.7),
        "last_verified": ts, "created_at": ts, "updated_at": ts,
    })
    return jsonify({"status": "created", "entity_key": entity_key})


@app.route("/wm/entity/list", methods=["GET"])
def wm_entity_list():
    db = get_db()
    etype = request.args.get("type", "")
    limit = safe_int(request.args.get("limit"), default=100, max_val=500)
    if etype:
        rows = db.execute(
            "SELECT entity_key, name, type, state, attributes, confidence, last_verified, updated_at FROM wm_entities WHERE type = ? ORDER BY confidence DESC LIMIT ?",
            [etype, limit]).fetchall()
    else:
        rows = db.execute(
            "SELECT entity_key, name, type, state, attributes, confidence, last_verified, updated_at FROM wm_entities ORDER BY updated_at DESC LIMIT ?",
            [limit]).fetchall()
    results = [{"entity_key": r[0], "name": r[1], "type": r[2], "state": r[3],
                "attributes": _safe_json(r[4], {}), "confidence": r[5],
                "last_verified": r[6], "updated_at": r[7]} for r in rows]
    return jsonify({"count": len(results), "entities": results})


@app.route("/wm/relation", methods=["POST"])
def wm_relation_create():
    db = get_db()
    data = request.json or {}
    subject = (data.get("subject") or "").strip()
    predicate = (data.get("predicate") or "").strip()
    obj = (data.get("object") or "").strip()
    if not (subject and predicate and obj):
        return jsonify({"error": "subject, predicate, object required"}), 400
    ts = now_iso()
    conf = clamp_float(data.get("confidence", 0.6), default=0.6)
    evidence = data.get("evidence", "")
    if not isinstance(evidence, str):
        evidence = json.dumps(evidence, ensure_ascii=False)
    prov = data.get("provenance", [])
    if not isinstance(prov, str):
        prov = json.dumps(prov, ensure_ascii=False)
    existing = db.execute(
        "SELECT id FROM wm_relations WHERE subject = ? AND predicate = ? AND object = ?",
        [subject, predicate, obj]).fetchone()
    if existing:
        db.execute(
            "UPDATE wm_relations SET confidence = ?, evidence = ?, provenance = ?, last_verified = ?, updated_at = ? WHERE id = ?",
            [conf, evidence, prov, ts, ts, existing[0]])
        return jsonify({"status": "updated", "id": existing[0]})
    db["wm_relations"].insert({
        "subject": subject, "predicate": predicate, "object": obj,
        "confidence": conf, "evidence": evidence, "provenance": prov,
        "last_verified": ts, "created_at": ts, "updated_at": ts,
    })
    return jsonify({"status": "created"})


@app.route("/wm/relation/list", methods=["GET"])
def wm_relation_list():
    db = get_db()
    subject = request.args.get("subject", "")
    predicate = request.args.get("predicate", "")
    limit = safe_int(request.args.get("limit"), default=100, max_val=500)
    q = "SELECT id, subject, predicate, object, confidence, evidence, provenance, last_verified, updated_at FROM wm_relations WHERE 1=1"
    args = []
    if subject:
        q += " AND subject = ?"
        args.append(subject)
    if predicate:
        q += " AND predicate = ?"
        args.append(predicate)
    q += " ORDER BY confidence DESC LIMIT ?"
    args.append(limit)
    rows = db.execute(q, args).fetchall()
    results = [{"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                "confidence": r[4], "evidence": r[5],
                "provenance": _safe_json(r[6], []),
                "last_verified": r[7], "updated_at": r[8]} for r in rows]
    return jsonify({"count": len(results), "relations": results})


@app.route("/wm/event", methods=["POST"])
def wm_event_create():
    db = get_db()
    data = request.json or {}
    etype = (data.get("event_type") or "").strip()
    if not etype:
        return jsonify({"error": "event_type required"}), 400
    ts = now_iso()
    def _s(v):
        return v if isinstance(v, str) else json.dumps(v or [], ensure_ascii=False)
    db["wm_events"].insert({
        "event_type": etype,
        "actor": data.get("actor", ""),
        "target": data.get("target", ""),
        "payload": _s(data.get("payload", {})),
        "causes": _s(data.get("causes", [])),
        "effects": _s(data.get("effects", [])),
        "occurred_at": data.get("occurred_at", ts),
        "created_at": ts,
    })
    return jsonify({"status": "created"})


@app.route("/wm/event/list", methods=["GET"])
def wm_event_list():
    db = get_db()
    etype = request.args.get("event_type", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    if etype:
        rows = db.execute(
            "SELECT id, event_type, actor, target, payload, causes, effects, occurred_at, created_at FROM wm_events WHERE event_type = ? ORDER BY occurred_at DESC LIMIT ?",
            [etype, limit]).fetchall()
    else:
        rows = db.execute(
            "SELECT id, event_type, actor, target, payload, causes, effects, occurred_at, created_at FROM wm_events ORDER BY occurred_at DESC LIMIT ?",
            [limit]).fetchall()
    results = [{"id": r[0], "event_type": r[1], "actor": r[2], "target": r[3],
                "payload": _safe_json(r[4], {}),
                "causes": _safe_json(r[5], []),
                "effects": _safe_json(r[6], []),
                "occurred_at": r[7], "created_at": r[8]} for r in rows]
    return jsonify({"count": len(results), "events": results})


def _compute_calibration_offset(db, days=30, min_samples=5):
    """Closed-loop calibration. Returns mean signed Brier-like error over recent
    resolved predictions. Positive = overconfident (lower future conf).
    Negative = underconfident (raise future conf)."""
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)).isoformat()
    rows = db.execute(
        "SELECT calibration FROM wm_predictions "
        "WHERE resolved=1 AND calibration IS NOT NULL AND resolved_at >= ?",
        [cutoff]).fetchall()
    n = len(rows)
    if n < min_samples:
        return {"offset": 0.0, "samples": n, "sufficient": False, "window_days": days}
    avg = sum(r[0] for r in rows) / n
    return {"offset": avg, "samples": n, "sufficient": True, "window_days": days}


@app.route("/calibration/current", methods=["GET"])
def calibration_current():
    """Returns the live calibration offset to apply to new predictions."""
    db = get_db()
    days = safe_int(request.args.get("days"), default=30, max_val=365)
    return jsonify(_compute_calibration_offset(db, days=days))


@app.route("/wm/prediction", methods=["POST"])
def wm_prediction_create():
    """Create a testable prediction. Body: {hypothesis, condition, predicted_outcome, counterfactual?, confidence?, due_at?}.
    v2.13: applies calibration offset from recently resolved predictions to produce
    confidence_adjusted (the forecast actually used). confidence_raw stores the
    user-supplied value for audit."""
    db = get_db()
    data = request.json or {}
    hypothesis = (data.get("hypothesis") or "").strip()
    outcome = (data.get("predicted_outcome") or "").strip()
    if not hypothesis or not outcome:
        return jsonify({"error": "hypothesis and predicted_outcome required"}), 400
    raw = clamp_float(data.get("confidence", 0.5), default=0.5)
    cal = _compute_calibration_offset(db)
    adjusted = max(0.05, min(0.95, raw - cal["offset"])) if cal["sufficient"] else raw
    ts = now_iso()
    db["wm_predictions"].insert({
        "hypothesis": hypothesis,
        "condition": data.get("condition", ""),
        "predicted_outcome": outcome,
        "counterfactual": data.get("counterfactual", ""),
        "confidence": adjusted,           # the forecast actually used (calibrated)
        "confidence_raw": raw,            # user-supplied original
        "confidence_adjusted": adjusted,  # mirror for analytics
        "due_at": data.get("due_at", ""),
        "resolved": 0,
        "actual_outcome": "",
        "resolved_at": "",
        "calibration": None,
        "created_at": ts,
        "updated_at": ts,
    })
    return jsonify({
        "status": "created",
        "confidence_raw": raw,
        "confidence_adjusted": adjusted,
        "calibration_offset_applied": cal["offset"] if cal["sufficient"] else 0.0,
        "calibration_samples": cal["samples"],
    })


@app.route("/wm/prediction/list", methods=["GET"])
def wm_prediction_list():
    db = get_db()
    resolved = request.args.get("resolved", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    cols = ("id, hypothesis, condition, predicted_outcome, counterfactual, "
            "confidence, due_at, resolved, actual_outcome, resolved_at, "
            "calibration, created_at, confidence_raw, confidence_adjusted")
    if resolved in ("1", "0"):
        rows = db.execute(
            f"SELECT {cols} FROM wm_predictions WHERE resolved = ? ORDER BY created_at DESC LIMIT ?",
            [int(resolved), limit]).fetchall()
    else:
        rows = db.execute(
            f"SELECT {cols} FROM wm_predictions ORDER BY created_at DESC LIMIT ?",
            [limit]).fetchall()
    results = [{"id": r[0], "hypothesis": r[1], "condition": r[2],
                "predicted_outcome": r[3], "counterfactual": r[4],
                "confidence": r[5], "due_at": r[6], "resolved": bool(r[7]),
                "actual_outcome": r[8], "resolved_at": r[9],
                "calibration": r[10], "created_at": r[11],
                "confidence_raw": r[12], "confidence_adjusted": r[13]} for r in rows]
    return jsonify({"count": len(results), "predictions": results})


@app.route("/wm/prediction/<int:pred_id>/resolve", methods=["PATCH"])
def wm_prediction_resolve(pred_id):
    """Resolve a prediction. Body: {actual_outcome, correct: true/false}"""
    db = get_db()
    row = db.execute(
        "SELECT id, confidence, predicted_outcome FROM wm_predictions WHERE id = ?",
        [pred_id]).fetchone()
    if not row:
        return jsonify({"error": f"Prediction {pred_id} not found"}), 404
    data = request.json or {}
    actual = data.get("actual_outcome", "")
    correct = bool(data.get("correct", False))
    conf = row[1] or 0.5
    calibration = (conf - (1.0 if correct else 0.0))  # Brier-like signed error
    ts = now_iso()
    db.execute(
        "UPDATE wm_predictions SET resolved = 1, actual_outcome = ?, resolved_at = ?, calibration = ?, updated_at = ? WHERE id = ?",
        [actual, ts, calibration, ts, pred_id])
    return jsonify({"status": "resolved", "id": pred_id, "calibration_gap": calibration})


# -- Verifier / Critic --

_VALID_CHECK_TYPES = {"factual", "consistency", "goal_alignment", "hallucination", "uncertainty", "evidence"}


@app.route("/verify", methods=["POST"])
def verify_create():
    """Record a verification. Body: {subject_type, subject_id, check_type, passed: bool,
       confidence?, reason, evidence?, sources?, halluc_risk?, required_evidence?}"""
    db = get_db()
    data = request.json or {}
    subject_type = (data.get("subject_type") or "").strip()
    subject_id = str(data.get("subject_id") or "").strip()
    check_type = (data.get("check_type") or "").strip()
    if not (subject_type and subject_id and check_type):
        return jsonify({"error": "subject_type, subject_id, check_type required"}), 400
    if check_type not in _VALID_CHECK_TYPES:
        return jsonify({"error": f"check_type must be one of {sorted(_VALID_CHECK_TYPES)}"}), 400
    ts = now_iso()
    def _s(v):
        return v if isinstance(v, str) else json.dumps(v or [], ensure_ascii=False)
    passed = bool(data.get("passed"))
    verifier_conf = clamp_float(data.get("confidence", 0.5), default=0.5)
    db["verifications"].insert({
        "subject_type": subject_type,
        "subject_id": subject_id,
        "check_type": check_type,
        "passed": 1 if passed else 0,
        "confidence": verifier_conf,
        "reason": data.get("reason", ""),
        "evidence": _s(data.get("evidence", "")),
        "sources": _s(data.get("sources", [])),
        "halluc_risk": clamp_float(data.get("halluc_risk", 0.0), default=0.0),
        "required_evidence": 1 if data.get("required_evidence") else 0,
        "created_at": ts,
    })
    # v2.13 closed loop: feed verification result back to the subject's
    # confidence + last_verified timestamp.
    confidence_delta = 0.0
    propagated = False
    try:
        sid_int = int(subject_id) if subject_id.isdigit() else None
    except Exception:
        sid_int = None
    if sid_int is not None and subject_type in ("memory", "entity"):
        table = "memories" if subject_type == "memory" else "entities"
        # Adjust confidence: pass → +0.5*verifier_conf*(1-current), fail → -0.5*verifier_conf*current
        # Both clamp to [0.05, 0.99] and update last_verified.
        row = db.execute(
            f"SELECT confidence FROM {table} WHERE id = ?", [sid_int]).fetchone()
        if row:
            current = row[0] if row[0] is not None else 0.8
            if passed:
                confidence_delta = 0.5 * verifier_conf * (1 - current)
            else:
                confidence_delta = -0.5 * verifier_conf * current
            new_conf = max(0.05, min(0.99, current + confidence_delta))
            db.execute(
                f"UPDATE {table} SET confidence = ?, last_verified = ? WHERE id = ?",
                [new_conf, ts, sid_int])
            propagated = True
    return jsonify({
        "status": "recorded",
        "feedback": {
            "propagated_to_subject": propagated,
            "confidence_delta": round(confidence_delta, 4),
        },
    })


@app.route("/verify/list", methods=["GET"])
def verify_list():
    db = get_db()
    subject_type = request.args.get("subject_type", "")
    subject_id = request.args.get("subject_id", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    q = "SELECT id, subject_type, subject_id, check_type, passed, confidence, reason, evidence, sources, halluc_risk, required_evidence, created_at FROM verifications WHERE 1=1"
    args = []
    if subject_type:
        q += " AND subject_type = ?"
        args.append(subject_type)
    if subject_id:
        q += " AND subject_id = ?"
        args.append(subject_id)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    rows = db.execute(q, args).fetchall()
    results = [{"id": r[0], "subject_type": r[1], "subject_id": r[2],
                "check_type": r[3], "passed": bool(r[4]), "confidence": r[5],
                "reason": r[6], "evidence": r[7],
                "sources": _safe_json(r[8], []),
                "halluc_risk": r[9], "required_evidence": bool(r[10]),
                "created_at": r[11]} for r in rows]
    return jsonify({"count": len(results), "verifications": results})


# -- Sandbox --

_VALID_SANDBOX_MODES = {"dry-run", "simulation", "live"}


@app.route("/sandbox/execute", methods=["POST"])
def sandbox_execute():
    """Record a sandboxed execution. Body: {mode, action, input?, simulated_output?,
       predicted_cost?, predicted_time_sec?, verdict?, plan_id?, goal_id?, skill_id?}"""
    db = get_db()
    data = request.json or {}
    mode = (data.get("mode") or "").lower().strip()
    if mode not in _VALID_SANDBOX_MODES:
        return jsonify({"error": f"mode must be one of {sorted(_VALID_SANDBOX_MODES)}"}), 400
    action = (data.get("action") or "").strip()
    if not action:
        return jsonify({"error": "action required"}), 400
    ts = now_iso()
    execution_id = data.get("execution_id") or f"x_{uuid.uuid4().hex[:8]}"
    def _s(v):
        return v if isinstance(v, str) else json.dumps(v or {}, ensure_ascii=False)
    db["sandbox_executions"].insert({
        "execution_id": execution_id,
        "mode": mode,
        "plan_id": data.get("plan_id", ""),
        "goal_id": data.get("goal_id", ""),
        "skill_id": safe_int(data.get("skill_id", 0), default=0, min_val=0, max_val=10**9),
        "action": action,
        "input": _s(data.get("input", {})),
        "simulated_output": _s(data.get("simulated_output", {})),
        "predicted_cost": clamp_float(data.get("predicted_cost", 0.0), default=0.0, lo=0.0, hi=10**6),
        "predicted_time_sec": clamp_float(data.get("predicted_time_sec", 0.0), default=0.0, lo=0.0, hi=10**6),
        "verdict": data.get("verdict", "pending"),
        "promoted_to_live": 1 if data.get("promoted_to_live") else 0,
        "created_at": ts,
    })
    # v2.13: closed-loop sandbox promotion check.
    # If this is a dry-run with verdict=ok and the previous N (default 3)
    # dry-runs of the same action all had verdict=ok → caller can safely
    # graduate to mode='live' next time. Returned as `auto_promotable`.
    auto_promotable = False
    promote_n = 3
    if mode == "dry-run" and data.get("verdict") == "ok":
        prev = db.execute(
            "SELECT verdict FROM sandbox_executions "
            "WHERE action = ? AND mode = 'dry-run' "
            "ORDER BY created_at DESC LIMIT ?",
            [action, promote_n]).fetchall()
        # Note: includes the row we just inserted, so all `promote_n` must be 'ok'.
        if len(prev) >= promote_n and all(r[0] == "ok" for r in prev):
            auto_promotable = True
    return jsonify({
        "status": "recorded",
        "execution_id": execution_id,
        "mode": mode,
        "auto_promotable": auto_promotable,
        "promote_threshold": promote_n,
    })


@app.route("/sandbox/<execution_id>/promote", methods=["POST"])
def sandbox_promote(execution_id):
    """Mark a dry-run execution as promoted_to_live=1. Use after the caller
    has actually executed the live action."""
    db = get_db()
    row = db.execute(
        "SELECT id, mode FROM sandbox_executions WHERE execution_id = ?",
        [execution_id]).fetchone()
    if not row:
        return jsonify({"error": "execution_id not found"}), 404
    db.execute(
        "UPDATE sandbox_executions SET promoted_to_live = 1 WHERE execution_id = ?",
        [execution_id])
    return jsonify({"status": "promoted", "execution_id": execution_id})


@app.route("/sandbox/can_live", methods=["GET"])
def sandbox_can_live():
    """Returns whether an action has accumulated enough successful dry-runs to
    be safely promoted to live mode. Query: ?action=<action>&n=<threshold>"""
    db = get_db()
    action = (request.args.get("action") or "").strip()
    n = safe_int(request.args.get("n"), default=3, min_val=1, max_val=20)
    if not action:
        return jsonify({"error": "action required"}), 400
    rows = db.execute(
        "SELECT verdict FROM sandbox_executions "
        "WHERE action = ? AND mode = 'dry-run' "
        "ORDER BY created_at DESC LIMIT ?",
        [action, n]).fetchall()
    can = len(rows) >= n and all(r[0] == "ok" for r in rows)
    return jsonify({
        "can_live": can,
        "action": action,
        "threshold": n,
        "samples": len(rows),
        "verdicts": [r[0] for r in rows],
    })


@app.route("/sandbox/list", methods=["GET"])
def sandbox_list():
    db = get_db()
    mode = request.args.get("mode", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    q = "SELECT id, execution_id, mode, plan_id, goal_id, skill_id, action, input, simulated_output, predicted_cost, predicted_time_sec, verdict, promoted_to_live, created_at FROM sandbox_executions WHERE 1=1"
    args = []
    if mode:
        q += " AND mode = ?"
        args.append(mode)
    q += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    rows = db.execute(q, args).fetchall()
    results = [{"id": r[0], "execution_id": r[1], "mode": r[2], "plan_id": r[3],
                "goal_id": r[4], "skill_id": r[5], "action": r[6],
                "input": _safe_json(r[7], {}),
                "simulated_output": _safe_json(r[8], {}),
                "predicted_cost": r[9], "predicted_time_sec": r[10],
                "verdict": r[11], "promoted_to_live": bool(r[12]),
                "created_at": r[13]} for r in rows]
    return jsonify({"count": len(results), "executions": results})


# -- Skill Compiler Upgrade --

_MATURITY_RANK = {"draft": 0, "beta": 1, "stable": 2, "deprecated": -1}


@app.route("/skill/<int:skill_id>/record", methods=["POST"])
def skill_record(skill_id):
    """Record a skill execution outcome. Body: {outcome, cost?, time_sec?, failure_domain?}"""
    db = get_db()
    row = db.execute(
        "SELECT id, success_count, failure_count, cost_avg, time_avg_sec, failure_domains FROM skills WHERE id = ?",
        [skill_id]).fetchone()
    if not row:
        return jsonify({"error": f"Skill {skill_id} not found"}), 404
    data = request.json or {}
    outcome = (data.get("outcome") or "").lower()
    if outcome not in ("success", "failure"):
        return jsonify({"error": "outcome must be 'success' or 'failure'"}), 400
    cost = float(data.get("cost", 0.0) or 0.0)
    time_sec = float(data.get("time_sec", 0.0) or 0.0)
    failure_domain = (data.get("failure_domain") or "").strip() if outcome == "failure" else None

    prev_s = row[1] or 0
    prev_f = row[2] or 0
    new_s = prev_s + (1 if outcome == "success" else 0)
    new_f = prev_f + (1 if outcome == "failure" else 0)
    total = new_s + new_f
    prev_total = prev_s + prev_f
    new_cost_avg = ((row[3] or 0.0) * prev_total + cost) / total if total else 0.0
    new_time_avg = ((row[4] or 0.0) * prev_total + time_sec) / total if total else 0.0
    domains = _safe_json(row[5], [])
    if failure_domain:
        domains = (domains + [failure_domain])[-20:]

    ts = now_iso()
    db.execute(
        """UPDATE skills SET success_count = ?, failure_count = ?, cost_avg = ?, time_avg_sec = ?,
           failure_domains = ?, times_used = times_used + 1, last_used = ? WHERE id = ?""",
        [new_s, new_f, new_cost_avg, new_time_avg,
         json.dumps(domains, ensure_ascii=False), ts, skill_id])
    return jsonify({"status": "recorded", "id": skill_id,
                    "success_count": new_s, "failure_count": new_f,
                    "success_rate": (new_s / total) if total else None})


@app.route("/skill/auto_promote", methods=["POST"])
def skill_auto_promote():
    """Closed-loop skill ratchet (v2.13). Runs the full graduation rules across
    all skills:
      draft   → beta       if success_count >= 1
      beta    → stable     if success_count >= 3 AND success_rate >= 0.66
      stable  → deprecated if failure_rate > 0.5 (over total runs)
    Returns a list of transitions made. Idempotent within a run."""
    db = get_db()
    rows = db.execute(
        "SELECT id, name, success_count, failure_count, maturity FROM skills"
    ).fetchall()
    transitions = []
    for r in rows:
        sid, name, sc, fc, maturity = r[0], r[1], r[2] or 0, r[3] or 0, r[4] or "draft"
        total = sc + fc
        sr = sc / total if total else 0.0
        new_maturity = None
        reason = None
        if maturity == "draft" and sc >= 1:
            new_maturity = "beta"
            reason = f"sc={sc} >= 1"
        elif maturity == "beta" and sc >= 3 and sr >= 0.66:
            new_maturity = "stable"
            reason = f"sc={sc} >= 3 AND sr={sr:.2f} >= 0.66"
        elif maturity == "stable" and total >= 5 and (fc / total) > 0.5:
            new_maturity = "deprecated"
            reason = f"failure_rate={fc/total:.2f} > 0.5 over {total} runs"
        if new_maturity:
            db.execute("UPDATE skills SET maturity = ? WHERE id = ?",
                       [new_maturity, sid])
            transitions.append({
                "skill_id": sid,
                "name": name,
                "from": maturity,
                "to": new_maturity,
                "reason": reason,
                "runs": total,
                "success_rate": round(sr, 3),
            })
    return jsonify({
        "status": "ratcheted",
        "transitions": transitions,
        "skills_evaluated": len(rows),
    })


@app.route("/skill/<int:skill_id>/promote", methods=["PATCH"])
def skill_promote(skill_id):
    """Promote skill maturity. Body: {maturity: 'draft'|'beta'|'stable'|'deprecated'}"""
    db = get_db()
    row = db.execute("SELECT id, success_count, failure_count, maturity FROM skills WHERE id = ?", [skill_id]).fetchone()
    if not row:
        return jsonify({"error": f"Skill {skill_id} not found"}), 404
    data = request.json or {}
    target = (data.get("maturity") or "").lower()
    if target not in _MATURITY_RANK:
        return jsonify({"error": f"maturity must be one of {list(_MATURITY_RANK)}"}), 400
    total = (row[1] or 0) + (row[2] or 0)
    sr = (row[1] or 0) / total if total else 0.0
    # Guardrails: need wins before promoting to stable
    if target == "stable" and (total < 3 or sr < 0.66):
        return jsonify({"error": f"cannot promote to stable: runs={total}, success_rate={sr:.2f} (need >=3 runs and >=0.66 success)"}), 400
    if target == "beta" and total < 1:
        return jsonify({"error": "cannot promote to beta without at least one recorded run"}), 400
    db.execute("UPDATE skills SET maturity = ? WHERE id = ?", [target, skill_id])
    return jsonify({"status": "promoted", "id": skill_id, "maturity": target,
                    "runs": total, "success_rate": sr})


# -- Experiment Engine --

@app.route("/experiment", methods=["POST"])
def experiment_create():
    """Create an experiment. Body: {hypothesis, context?, metric, min_delta?, min_samples?, variants: [{name, description}]}"""
    db = get_db()
    data = request.json or {}
    hypothesis = (data.get("hypothesis") or "").strip()
    metric = (data.get("metric") or "").strip()
    variants = data.get("variants") or []
    if not hypothesis or not metric:
        return jsonify({"error": "hypothesis and metric required"}), 400
    if not isinstance(variants, list) or len(variants) < 2:
        return jsonify({"error": "at least 2 variants required"}), 400
    ts = now_iso()
    exp_id = data.get("experiment_id") or f"e_{uuid.uuid4().hex[:8]}"
    db["experiments"].insert({
        "experiment_id": exp_id,
        "hypothesis": hypothesis,
        "context": data.get("context", ""),
        "metric": metric,
        "min_delta": clamp_float(data.get("min_delta", 0.05), default=0.05, lo=0.0, hi=1.0),
        "min_samples": safe_int(data.get("min_samples", 10), default=10, min_val=2, max_val=10**6),
        "variants": json.dumps(variants, ensure_ascii=False),
        "observations": json.dumps([], ensure_ascii=False),
        "status": "running",
        "winner": "",
        "conclusion": "",
        "started_at": ts,
        "concluded_at": "",
        "created_at": ts,
        "updated_at": ts,
    })
    return jsonify({"status": "created", "experiment_id": exp_id})


@app.route("/experiment/list", methods=["GET"])
def experiment_list():
    db = get_db()
    status = request.args.get("status", "")
    limit = safe_int(request.args.get("limit"), default=50, max_val=500)
    if status:
        rows = db.execute(
            "SELECT id, experiment_id, hypothesis, context, metric, min_delta, min_samples, variants, observations, status, winner, conclusion, started_at, concluded_at FROM experiments WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
            [status, limit]).fetchall()
    else:
        rows = db.execute(
            "SELECT id, experiment_id, hypothesis, context, metric, min_delta, min_samples, variants, observations, status, winner, conclusion, started_at, concluded_at FROM experiments ORDER BY updated_at DESC LIMIT ?",
            [limit]).fetchall()
    results = []
    for r in rows:
        obs = _safe_json(r[8], [])
        variants = _safe_json(r[7], [])
        per_variant = {}
        for o in obs:
            v = o.get("variant")
            if not v:
                continue
            slot = per_variant.setdefault(v, {"n": 0, "sum": 0.0})
            slot["n"] += 1
            slot["sum"] += float(o.get("value", 0.0) or 0.0)
        summary = {v: {"n": d["n"], "mean": (d["sum"] / d["n"]) if d["n"] else None}
                   for v, d in per_variant.items()}
        results.append({
            "id": r[0], "experiment_id": r[1], "hypothesis": r[2], "context": r[3],
            "metric": r[4], "min_delta": r[5], "min_samples": r[6],
            "variants": variants, "observations_count": len(obs),
            "per_variant_summary": summary,
            "status": r[9], "winner": r[10], "conclusion": r[11],
            "started_at": r[12], "concluded_at": r[13],
        })
    return jsonify({"count": len(results), "experiments": results})


@app.route("/experiment/<exp_id>/variant", methods=["POST"])
def experiment_add_variant(exp_id):
    db = get_db()
    row = db.execute("SELECT id, variants FROM experiments WHERE experiment_id = ?", [exp_id]).fetchone()
    if not row:
        return jsonify({"error": f"Experiment {exp_id} not found"}), 404
    data = request.json or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name required"}), 400
    variants = _safe_json(row[1], [])
    variants.append({"name": name, "description": data.get("description", "")})
    db.execute("UPDATE experiments SET variants = ?, updated_at = ? WHERE experiment_id = ?",
               [json.dumps(variants, ensure_ascii=False), now_iso(), exp_id])
    return jsonify({"status": "added", "variants": variants})


@app.route("/experiment/<exp_id>/observation", methods=["POST"])
def experiment_add_observation(exp_id):
    """Record an observation. Body: {variant, value, context?}"""
    db = get_db()
    row = db.execute("SELECT id, observations FROM experiments WHERE experiment_id = ?", [exp_id]).fetchone()
    if not row:
        return jsonify({"error": f"Experiment {exp_id} not found"}), 404
    data = request.json or {}
    variant = (data.get("variant") or "").strip()
    value = data.get("value")
    if not variant or value is None:
        return jsonify({"error": "variant and value required"}), 400
    obs = _safe_json(row[1], [])
    obs.append({"variant": variant, "value": float(value),
                "context": data.get("context", ""), "at": now_iso()})
    db.execute("UPDATE experiments SET observations = ?, updated_at = ? WHERE experiment_id = ?",
               [json.dumps(obs, ensure_ascii=False), now_iso(), exp_id])
    return jsonify({"status": "recorded", "observations": len(obs)})


@app.route("/experiment/<exp_id>/conclude", methods=["PATCH"])
def experiment_conclude(exp_id):
    """Close experiment. Auto-picks winner if enough samples and delta exceeds threshold,
    unless body passes {winner, conclusion}."""
    db = get_db()
    row = db.execute(
        "SELECT id, variants, observations, min_delta, min_samples FROM experiments WHERE experiment_id = ?",
        [exp_id]).fetchone()
    if not row:
        return jsonify({"error": f"Experiment {exp_id} not found"}), 404
    data = request.json or {}
    observations = _safe_json(row[2], [])
    min_delta = row[3] or 0.05
    min_samples = row[4] or 10
    per_variant = {}
    for o in observations:
        v = o.get("variant")
        if not v:
            continue
        slot = per_variant.setdefault(v, {"n": 0, "sum": 0.0})
        slot["n"] += 1
        slot["sum"] += float(o.get("value", 0.0) or 0.0)
    summary = {v: (d["sum"] / d["n"]) if d["n"] else None for v, d in per_variant.items()}

    winner = (data.get("winner") or "").strip()
    conclusion = data.get("conclusion") or ""
    if not winner:
        ranked = sorted(
            [(v, m) for v, m in summary.items() if m is not None],
            key=lambda x: x[1], reverse=True,
        )
        if len(ranked) >= 2:
            top_v, top_m = ranked[0]
            second_m = ranked[1][1]
            enough_samples = all(per_variant[v]["n"] >= min_samples for v, _ in ranked[:2])
            if enough_samples and (top_m - second_m) >= min_delta:
                winner = top_v
                conclusion = f"auto: {top_v} beat runner-up by {top_m - second_m:.3f} (>= {min_delta})"
            else:
                conclusion = conclusion or f"auto: inconclusive (delta={top_m - second_m:.3f}, samples={[per_variant[v]['n'] for v,_ in ranked[:2]]})"
        else:
            conclusion = conclusion or "auto: not enough variants with data"

    ts = now_iso()
    db.execute(
        "UPDATE experiments SET status = 'concluded', winner = ?, conclusion = ?, concluded_at = ?, updated_at = ? WHERE experiment_id = ?",
        [winner, conclusion, ts, ts, exp_id])
    return jsonify({"status": "concluded", "winner": winner or None,
                    "conclusion": conclusion, "summary": summary})


# -- Metrics Framework --

_KNOWN_METRICS = {
    "tasks_solved_no_correction_pct",
    "hallucination_rate",
    "time_to_complete_goal_sec",
    "skill_reuse_rate",
    "skill_success_rate",
    "world_model_precision",
    "calibration_gap",
    "actions_reverted_pct",
    "cost_per_useful_task",
    "goals_completed_per_week",
    "approved_improvements_effective_pct",
}


@app.route("/metric", methods=["POST"])
def metric_record():
    """Record a metric sample. Body: {name, value, unit?, context?, tags?, timestamp?}"""
    db = get_db()
    data = request.json or {}
    name = (data.get("name") or "").strip()
    value = data.get("value")
    if not name or value is None:
        return jsonify({"error": "name and value required"}), 400
    try:
        value = float(value)
    except (TypeError, ValueError):
        return jsonify({"error": "value must be numeric"}), 400
    tags = data.get("tags", [])
    if not isinstance(tags, str):
        tags = json.dumps(tags, ensure_ascii=False)
    db["metrics"].insert({
        "name": name,
        "value": value,
        "unit": data.get("unit", ""),
        "context": data.get("context", ""),
        "tags": tags,
        "timestamp": data.get("timestamp") or now_iso(),
    })
    return jsonify({"status": "recorded", "name": name, "value": value,
                    "known": name in _KNOWN_METRICS})


@app.route("/metric/list", methods=["GET"])
def metric_list():
    db = get_db()
    name = request.args.get("name", "")
    limit = safe_int(request.args.get("limit"), default=100, max_val=2000)
    if name:
        rows = db.execute(
            "SELECT id, name, value, unit, context, tags, timestamp FROM metrics WHERE name = ? ORDER BY timestamp DESC LIMIT ?",
            [name, limit]).fetchall()
    else:
        rows = db.execute(
            "SELECT id, name, value, unit, context, tags, timestamp FROM metrics ORDER BY timestamp DESC LIMIT ?",
            [limit]).fetchall()
    results = [{"id": r[0], "name": r[1], "value": r[2], "unit": r[3],
                "context": r[4], "tags": _safe_json(r[5], []),
                "timestamp": r[6]} for r in rows]
    return jsonify({"count": len(results), "samples": results})


@app.route("/metric/summary", methods=["GET"])
def metric_summary():
    """Aggregate latest value + 7-day trend per metric name."""
    db = get_db()
    names = db.execute("SELECT DISTINCT name FROM metrics").fetchall()
    now = datetime.datetime.now(datetime.timezone.utc)
    week_ago = (now - datetime.timedelta(days=7)).isoformat()
    out = []
    for (n,) in names:
        latest = db.execute(
            "SELECT value, timestamp FROM metrics WHERE name = ? ORDER BY timestamp DESC LIMIT 1",
            [n]).fetchone()
        agg = db.execute(
            "SELECT COUNT(*), AVG(value), MIN(value), MAX(value) FROM metrics WHERE name = ? AND timestamp >= ?",
            [n, week_ago]).fetchone()
        out.append({
            "name": n,
            "known": n in _KNOWN_METRICS,
            "latest": {"value": latest[0] if latest else None,
                       "timestamp": latest[1] if latest else None},
            "week": {"samples": agg[0], "avg": agg[1], "min": agg[2], "max": agg[3]},
        })
    out.sort(key=lambda x: x["name"])
    return jsonify({"count": len(out), "known_catalog": sorted(_KNOWN_METRICS), "metrics": out})


# -- Active Crons snapshot --

@app.route("/cron/active", methods=["POST"])
def cron_active_upsert():
    """Replace the snapshot of currently scheduled crons.
    Body: {crons: [{job_id, label, cron_expr, prompt_preview?}]}
    Any job_id not present in the payload is removed from the snapshot."""
    db = get_db()
    data = request.json or {}
    items = data.get("crons") or []
    ts = now_iso()
    seen = set()
    for c in items:
        jid = (c.get("job_id") or "").strip()
        if not jid:
            continue
        seen.add(jid)
        row = {
            "job_id": jid,
            "label": c.get("label", ""),
            "cron_expr": c.get("cron_expr", ""),
            "prompt_preview": (c.get("prompt_preview") or "")[:500],
            "registered_at": c.get("registered_at", ts),
            "updated_at": ts,
        }
        existing = db.execute("SELECT id FROM active_crons WHERE job_id = ?", [jid]).fetchone()
        if existing:
            db.execute(
                "UPDATE active_crons SET label = ?, cron_expr = ?, prompt_preview = ?, updated_at = ? WHERE job_id = ?",
                [row["label"], row["cron_expr"], row["prompt_preview"], ts, jid])
        else:
            db["active_crons"].insert(row)
    # Remove stale
    all_jids = [r[0] for r in db.execute("SELECT job_id FROM active_crons").fetchall()]
    for jid in all_jids:
        if jid not in seen:
            db.execute("DELETE FROM active_crons WHERE job_id = ?", [jid])
    return jsonify({"status": "synced", "count": len(seen), "removed": len(all_jids) - len(seen & set(all_jids))})


@app.route("/cron/active", methods=["GET"])
def cron_active_list():
    db = get_db()
    rows = db.execute(
        "SELECT job_id, label, cron_expr, prompt_preview, registered_at, updated_at FROM active_crons ORDER BY cron_expr"
    ).fetchall()
    results = [{"job_id": r[0], "label": r[1], "cron_expr": r[2],
                "prompt_preview": r[3], "registered_at": r[4], "updated_at": r[5]} for r in rows]
    return jsonify({"count": len(results), "crons": results})


_CRON_PROMPTS_PATH = Path.home() / ".claude" / "cron-prompts.md"


@app.route("/cron/prompts", methods=["GET"])
def cron_prompts_read():
    """Return ~/.claude/cron-prompts.md parsed into sections."""
    if not _CRON_PROMPTS_PATH.exists():
        return jsonify({"error": "cron-prompts.md not found", "path": str(_CRON_PROMPTS_PATH)}), 404
    text = _CRON_PROMPTS_PATH.read_text()
    sections = []
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            if current:
                sections.append(current)
            header = line[3:].strip()
            cron_match = re.search(r"`([^`]+)`", header)
            title = re.sub(r"—.*$", "", header).strip()
            current = {
                "title": title,
                "header": header,
                "cron_expr": cron_match.group(1) if cron_match else "",
                "body": "",
            }
        elif current is not None and not line.startswith("#"):
            current["body"] += line + "\n"
    if current:
        sections.append(current)
    for s in sections:
        s["body"] = s["body"].strip()
    return jsonify({"count": len(sections), "path": str(_CRON_PROMPTS_PATH),
                    "sections": sections})


# ───────────────────────────────────────────────────────────────
# Claude Code usage dashboard
# Spawns `claude` in a pty, sends /usage, parses the TUI output.
# Blocking ~25-35s per call. UI triggers on demand via refresh button.
# ───────────────────────────────────────────────────────────────
_USAGE_CACHE = {"data": None, "fetched_at": None, "fetching": False}
_USAGE_LOCK = threading.Lock()


def _find_claude_bin():
    import shutil
    candidates = [
        str(Path.home() / ".local/bin/claude"),
        str(Path.home() / ".claude/local/claude"),
        "/usr/local/bin/claude",
        "/usr/bin/claude",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return shutil.which("claude")


_ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(s):
    return _ANSI_RE.sub("", s)


def _parse_usage(raw):
    clean = re.sub(r"\s+", " ", _strip_ansi(raw))
    data = {}

    m = re.search(r"Current\s*session\s*[█▌░\s]*(\d+)\s*%\s*used\s*Rese?t?s?\s*s?\s*(.+?)(?=Current|Extra|Esc|What|$)", clean, re.I)
    if m:
        data["session"] = {"pct": int(m.group(1)), "resets": m.group(2).strip()}

    m = re.search(r"Current\s*week\s*\(all\s*models?\)\s*[█▌░\s]*(\d+)\s*%\s*used\s*Rese?t?s?\s*(.+?)(?=Current|Extra|Esc|What|$)", clean, re.I)
    if m:
        data["weekAll"] = {"pct": int(m.group(1)), "resets": m.group(2).strip()}

    m = re.search(r"Current\s*week\s*\(Sonnet\s*only\)\s*[█▌░\s]*(\d+)\s*%\s*used", clean, re.I)
    if m:
        data["weekSonnet"] = {"pct": int(m.group(1))}

    m = re.search(r"Extra\s*usage\s*[█▌░▏\s]*(\d+)\s*%\s*used\s*\$?\s*([\d.]+)\s*\/\s*\$?\s*([\d.]+)\s*spent\s*·?\s*Rese?t?s?\s*(.+?)(?=Esc|Last|$)", clean, re.I)
    if m:
        data["extra"] = {
            "pct": int(m.group(1)),
            "spent": m.group(2),
            "total": m.group(3),
            "resets": m.group(4).strip(),
        }

    m = re.search(r"(\d+)\s*%\s*of\s*your\s*usage\s*was\s*while\s*(\d+\+?\s*sessions?\s*ran\s*in\s*parallel)", clean, re.I)
    if m:
        data["insight"] = f"{m.group(1)}% of your usage was while {m.group(2)}"

    return data if (data.get("session") or data.get("weekAll") or data.get("extra")) else None


def _fetch_claude_usage():
    """Spawn `claude` via pexpect, send /usage, harvest the TUI output."""
    try:
        import pexpect
    except ImportError:
        return {"error": "pexpect not installed"}

    binpath = _find_claude_bin()
    if not binpath:
        return {"error": "claude binary not found"}

    buf = ""
    child = None
    # Pick a dir claude already trusts to skip the "do you trust this folder" prompt.
    # We read ~/.claude.json to find one.
    cwd = str(Path.home())
    try:
        with open(Path.home() / ".claude.json") as f:
            _cfg = json.load(f)
        trusted_dirs = [p for p, v in _cfg.get("projects", {}).items()
                        if isinstance(v, dict) and v.get("hasTrustDialogAccepted") and os.path.isdir(p)]
        if trusted_dirs:
            cwd = trusted_dirs[0]
    except Exception:
        pass
    try:
        print(f"[usage] spawning {binpath} from cwd={cwd}", flush=True)
        child = pexpect.spawn(
            binpath, dimensions=(50, 120), encoding="utf-8", timeout=None,
            cwd=cwd,
            env={**os.environ, "TERM": "xterm-256color", "PWD": cwd, "HOME": str(Path.home())},
        )

        def read_avail(timeout=1.0):
            try:
                return child.read_nonblocking(size=100_000, timeout=timeout)
            except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF):
                return ""
            except Exception:
                return ""

        # Wait for splash then auto-confirm the trust prompt if it appears.
        trust_confirmed = False
        deadline = time.time() + 12.0
        while time.time() < deadline:
            chunk = read_avail(0.5)
            if chunk:
                buf += chunk
                if not trust_confirmed and ("trust this folder" in buf.lower() or "Enter to confirm" in buf):
                    child.send("\r")  # press Enter to accept "1. Yes"
                    trust_confirmed = True
                    time.sleep(2.5)

        # Type /usage then Enter. claude TUI treats \r as submit;
        # sendline() sends \n which triggers autocomplete instead of submit.
        child.send("/usage")
        time.sleep(0.5)
        child.send("\r")

        deadline = time.time() + 18.0
        while time.time() < deadline:
            chunk = read_avail(0.6)
            if chunk:
                buf += chunk

        try:
            child.send("\x1b")
            time.sleep(0.5)
            child.sendline("/exit")
            time.sleep(1.5)
        except Exception:
            pass
    finally:
        if child:
            try:
                child.terminate(force=True)
            except Exception:
                pass

    data = _parse_usage(buf)
    if not data:
        return {"error": "no usage data parsed", "raw_preview": _strip_ansi(buf)[-400:]}
    return {"ok": True, "data": data}


def _fetch_usage_background():
    with _USAGE_LOCK:
        if _USAGE_CACHE["fetching"]:
            return
        _USAGE_CACHE["fetching"] = True
    try:
        result = _fetch_claude_usage()
        _USAGE_CACHE["data"] = result
        _USAGE_CACHE["fetched_at"] = now_iso()
    finally:
        _USAGE_CACHE["fetching"] = False


@app.route("/usage/claude", methods=["GET"])
def usage_claude_get():
    return jsonify({
        "fetched_at": _USAGE_CACHE["fetched_at"],
        "fetching": _USAGE_CACHE["fetching"],
        "result": _USAGE_CACHE["data"],
    })


@app.route("/usage/claude/refresh", methods=["POST"])
def usage_claude_refresh():
    sync = request.args.get("sync") in ("1", "true")
    if sync:
        _fetch_usage_background()
        return jsonify({
            "fetched_at": _USAGE_CACHE["fetched_at"],
            "fetching": _USAGE_CACHE["fetching"],
            "result": _USAGE_CACHE["data"],
        })
    t = threading.Thread(target=_fetch_usage_background, daemon=True)
    t.start()
    return jsonify({"started": True, "fetching": True})


# ───────────────────────────────────────────────────────────────
# Codex usage dashboard
# Spawns `codex` in a pty, sends /status, parses the panel.
# Mirrors Claude's /usage/claude approach.
# ───────────────────────────────────────────────────────────────
_CODEX_USAGE_CACHE = {"data": None, "fetched_at": None, "fetching": False}
_CODEX_USAGE_LOCK = threading.Lock()


def _find_codex_bin():
    import shutil
    candidates = [
        str(Path.home() / ".npm-global/bin/codex"),
        str(Path.home() / ".local/bin/codex"),
        "/usr/local/bin/codex",
        "/usr/bin/codex",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return shutil.which("codex")


def _parse_codex(raw):
    clean = re.sub(r"\s+", " ", _strip_ansi(raw))
    data = {}

    m = re.search(r"5h\s*limit\s*:?\s*\[[█░▌▏\s]*\]\s*(\d+)\s*%\s*left\s*\(\s*resets?\s+([^)]+?)\)", clean, re.I)
    if m:
        pct_left = int(m.group(1))
        data["session5h"] = {
            "pct_used": max(0, min(100, 100 - pct_left)),
            "pct_left": pct_left,
            "resets": m.group(2).strip(),
        }

    m = re.search(r"Weekly\s*limit\s*:?\s*\[[█░▌▏\s]*\]\s*(\d+)\s*%\s*left\s*\(\s*resets?\s+([^)]+?)\)", clean, re.I)
    if m:
        pct_left = int(m.group(1))
        data["weekly"] = {
            "pct_used": max(0, min(100, 100 - pct_left)),
            "pct_left": pct_left,
            "resets": m.group(2).strip(),
        }

    m = re.search(r"Account\s*:?\s*(\S+?@\S+?)\s*\(([^)]+)\)", clean, re.I)
    if m:
        data["account"] = {"email": m.group(1).strip(), "plan": m.group(2).strip()}

    m = re.search(r"Model\s*:?\s*([^\s│()]+)\s+\(([^)]+)\)", clean, re.I)
    if m:
        data["model"] = {"name": m.group(1).strip(), "detail": m.group(2).strip()}

    return data if (data.get("session5h") or data.get("weekly")) else None


def _fetch_codex_usage():
    """Spawn `codex` via pexpect, send /status, harvest the panel."""
    try:
        import pexpect
    except ImportError:
        return {"error": "pexpect not installed"}

    binpath = _find_codex_bin()
    if not binpath:
        return {"error": "codex binary not found"}

    workdir = Path.home() / ".codex" / ".tmp" / "status-workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    buf = ""
    child = None
    try:
        child = pexpect.spawn(
            binpath, dimensions=(60, 140), encoding="utf-8", timeout=None,
            cwd=str(workdir),
            env={**os.environ, "TERM": "xterm-256color",
                 "PWD": str(workdir), "HOME": str(Path.home())},
        )

        def read_avail(timeout=0.5):
            try:
                return child.read_nonblocking(size=100_000, timeout=timeout)
            except (pexpect.exceptions.TIMEOUT, pexpect.exceptions.EOF):
                return ""
            except Exception:
                return ""

        # Let the REPL boot
        deadline = time.time() + 8.0
        while time.time() < deadline:
            chunk = read_avail(0.4)
            if chunk:
                buf += chunk

        child.send("/status")
        time.sleep(1.0)
        child.send("\r")
        time.sleep(0.5)
        child.send("\n")

        deadline = time.time() + 12.0
        while time.time() < deadline:
            chunk = read_avail(0.4)
            if chunk:
                buf += chunk
    finally:
        if child:
            try:
                child.terminate(force=True)
            except Exception:
                pass

    data = _parse_codex(buf)
    if not data:
        return {"error": "no codex status parsed", "raw_preview": _strip_ansi(buf)[-400:]}
    return {"ok": True, "data": data}


def _fetch_codex_usage_background():
    with _CODEX_USAGE_LOCK:
        if _CODEX_USAGE_CACHE["fetching"]:
            return
        _CODEX_USAGE_CACHE["fetching"] = True
    try:
        result = _fetch_codex_usage()
        _CODEX_USAGE_CACHE["data"] = result
        _CODEX_USAGE_CACHE["fetched_at"] = now_iso()
    finally:
        _CODEX_USAGE_CACHE["fetching"] = False


@app.route("/usage/codex", methods=["GET"])
def usage_codex_get():
    return jsonify({
        "fetched_at": _CODEX_USAGE_CACHE["fetched_at"],
        "fetching": _CODEX_USAGE_CACHE["fetching"],
        "result": _CODEX_USAGE_CACHE["data"],
    })


@app.route("/usage/codex/refresh", methods=["POST"])
def usage_codex_refresh():
    sync = request.args.get("sync") in ("1", "true")
    if sync:
        _fetch_codex_usage_background()
        return jsonify({
            "fetched_at": _CODEX_USAGE_CACHE["fetched_at"],
            "fetching": _CODEX_USAGE_CACHE["fetching"],
            "result": _CODEX_USAGE_CACHE["data"],
        })
    t = threading.Thread(target=_fetch_codex_usage_background, daemon=True)
    t.start()
    return jsonify({"started": True, "fetching": True})


# ───────────────────────────────────────────────────────────────
# ElevenLabs usage dashboard
# Plain REST call to /v1/user/subscription with xi-api-key.
# ───────────────────────────────────────────────────────────────
_ELEVEN_USAGE_CACHE = {"data": None, "fetched_at": None, "fetching": False}
_ELEVEN_USAGE_LOCK = threading.Lock()


def _fetch_elevenlabs_usage():
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        return {"error": "ELEVENLABS_API_KEY not configured"}
    try:
        import urllib.request
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": api_key, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": f"fetch failed: {e}"}

    tier = payload.get("tier", "N/A")
    limit = int(payload.get("character_limit", 0) or 0)
    used = int(payload.get("character_count", 0) or 0)
    reset_unix = payload.get("next_character_count_reset_unix", 0) or 0
    remaining = max(0, limit - used)
    pct = round((used * 100 / limit), 1) if limit > 0 else 0.0
    reset_iso = ""
    try:
        if reset_unix:
            reset_iso = datetime.datetime.fromtimestamp(int(reset_unix), tz=datetime.timezone.utc).isoformat()
    except Exception:
        reset_iso = ""

    return {"ok": True, "data": {
        "tier": tier,
        "characters_used": used,
        "characters_limit": limit,
        "characters_remaining": remaining,
        "pct_used": pct,
        "reset_at": reset_iso,
    }}


def _fetch_elevenlabs_background():
    with _ELEVEN_USAGE_LOCK:
        if _ELEVEN_USAGE_CACHE["fetching"]:
            return
        _ELEVEN_USAGE_CACHE["fetching"] = True
    try:
        result = _fetch_elevenlabs_usage()
        _ELEVEN_USAGE_CACHE["data"] = result
        _ELEVEN_USAGE_CACHE["fetched_at"] = now_iso()
    finally:
        _ELEVEN_USAGE_CACHE["fetching"] = False


@app.route("/usage/elevenlabs", methods=["GET"])
def usage_elevenlabs_get():
    return jsonify({
        "fetched_at": _ELEVEN_USAGE_CACHE["fetched_at"],
        "fetching": _ELEVEN_USAGE_CACHE["fetching"],
        "result": _ELEVEN_USAGE_CACHE["data"],
    })


@app.route("/usage/elevenlabs/refresh", methods=["POST"])
def usage_elevenlabs_refresh():
    sync = request.args.get("sync") in ("1", "true")
    if sync:
        _fetch_elevenlabs_background()
        return jsonify({
            "fetched_at": _ELEVEN_USAGE_CACHE["fetched_at"],
            "fetching": _ELEVEN_USAGE_CACHE["fetching"],
            "result": _ELEVEN_USAGE_CACHE["data"],
        })
    t = threading.Thread(target=_fetch_elevenlabs_background, daemon=True)
    t.start()
    return jsonify({"started": True, "fetching": True})


if __name__ == "__main__":
    init_db()
    init_embeddings_table()
    _seed_keywords()
    _seed_autonomy_levels()
    _seed_capabilities()
    app.run(host="0.0.0.0", port=PORT)
