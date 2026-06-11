#!/usr/bin/env python3
"""memory_sync.py — sync bidireccional entre file-memories y la DB de Friday.

Direcciones:
  files -> DB : memorias .md curadas que no existen en la DB (match por name)
                se importan via POST /memory (genera embedding).
  DB -> mirror: TODAS las memorias de la DB se espejan en memory/db-mirror/
                como .md regenerables. Nunca toca los .md curados de nivel
                superior ni borra nada.

Garantías: cero deletes, cero overwrites de archivos curados. Si un name
existe en ambos lados con contenido distinto, se reporta en `conflicts`
(no se pisa ninguno).

Uso: python3 memory_sync.py [--dry-run]
"""
import json
import os
import re
import sqlite3
import sys
import urllib.request

API = "http://127.0.0.1:7777"
MEM_DIR = os.path.expanduser("~/.claude/projects/-home-br1/memory")
MIRROR_DIR = os.path.join(MEM_DIR, "db-mirror")
DB_PATH = os.environ.get("FRIDAY_DB_PATH", os.path.expanduser("~/.claude/memory.db"))

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_md(path):
    raw = open(path, encoding="utf-8").read()
    m = FRONTMATTER_RE.match(raw)
    meta, body = {}, raw
    if m:
        body = raw[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
    mtype = meta.get("type", "")
    return {
        "name": name,
        "type": mtype if mtype in ("user", "feedback", "project", "reference") else "reference",
        "description": meta.get("description", ""),
        "content": body.strip(),
    }


def api_post(path, payload):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120] or "unnamed"


def main():
    dry = "--dry-run" in sys.argv
    report = {"imported_to_db": [], "mirrored": 0, "conflicts": [], "errors": []}

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    db_rows = con.execute(
        "SELECT id, name, type, description, content, updated_at FROM memories").fetchall()
    db_by_name = {r["name"]: r for r in db_rows}
    db_contents = {re.sub(r"\s+", " ", (r["content"] or "").strip()) for r in db_rows}

    # files -> DB
    files = [f for f in sorted(os.listdir(MEM_DIR))
             if f.endswith(".md") and f != "MEMORY.md"
             and os.path.isfile(os.path.join(MEM_DIR, f))]
    for f in files:
        try:
            mem = parse_md(os.path.join(MEM_DIR, f))
        except Exception as e:
            report["errors"].append(f"parse {f}: {e}")
            continue
        existing = db_by_name.get(mem["name"])
        norm_content = re.sub(r"\s+", " ", mem["content"])
        if existing is None and norm_content in db_contents:
            report.setdefault("skipped_same_content", []).append(mem["name"])
            continue
        if existing is None:
            if not dry:
                try:
                    api_post("/memory", mem)
                except Exception as e:
                    report["errors"].append(f"import {mem['name']}: {e}")
                    continue
            report["imported_to_db"].append(mem["name"])
        elif (existing["content"] or "").strip() != mem["content"]:
            report["conflicts"].append(mem["name"])

    # DB -> mirror (regenerable, nunca toca los curados)
    if not dry:
        os.makedirs(MIRROR_DIR, exist_ok=True)
        readme = os.path.join(MIRROR_DIR, "README.md")
        if not os.path.exists(readme):
            open(readme, "w", encoding="utf-8").write(
                "# db-mirror\n\nEspejo regenerable de las memorias de la DB "
                "(memory.db), generado por memory_sync.py. NO editar a mano: "
                "los cambios van a la DB via la API. Este directorio NO se "
                "indexa en MEMORY.md.\n")
    # re-leer post-import para que el mirror incluya lo recién importado
    db_rows = con.execute(
        "SELECT id, name, type, description, content, updated_at FROM memories").fetchall()
    for r in db_rows:
        fname = f"{safe_filename(r['name'])}.md"
        out = (f"---\nid: {r['id']}\nname: {r['name']}\n"
               f"description: {r['description'] or ''}\n"
               f"type: {r['type'] or 'reference'}\n"
               f"updated_at: {r['updated_at'] or ''}\n---\n\n{r['content'] or ''}\n")
        if not dry:
            open(os.path.join(MIRROR_DIR, fname), "w", encoding="utf-8").write(out)
        report["mirrored"] += 1

    report["db_total_after"] = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
