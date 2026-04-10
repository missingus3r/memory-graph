#!/usr/bin/env bash
# Friday Memory CLI — wrapper for the memory HTTP API
# Usage:
#   memory.sh stats
#   memory.sh recall <topic>
#   memory.sh search <query>
#   memory.sh store <name> <type> <content>
#   memory.sh log <role> <content> [channel]
#   memory.sh list [type]
#   memory.sh recent [limit]
#   memory.sh entity-store <name> <type> <details_json>
#   memory.sh entity-search [query]
#   memory.sh conversation-search <query>

BASE="http://127.0.0.1:7777"

case "${1}" in
  stats)
    curl -s "$BASE/stats" | python3 -m json.tool
    ;;
  health)
    curl -s "$BASE/health"
    ;;
  recall)
    curl -s "$BASE/memory/recall?topic=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${2}'))")&limit=${3:-10}" | python3 -m json.tool
    ;;
  search)
    curl -s "$BASE/memory/search?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${2}'))")&type=${3}&limit=${4:-10}" | python3 -m json.tool
    ;;
  store)
    curl -s -X POST "$BASE/memory" -H "Content-Type: application/json" \
      -d "{\"name\": \"$2\", \"type\": \"$3\", \"content\": \"$4\", \"description\": \"$5\"}" | python3 -m json.tool
    ;;
  list)
    curl -s "$BASE/memory/list?type=${2}&limit=${3:-50}" | python3 -m json.tool
    ;;
  delete)
    curl -s -X DELETE "$BASE/memory/${2}" | python3 -m json.tool
    ;;
  log)
    curl -s -X POST "$BASE/conversation/log" -H "Content-Type: application/json" \
      -d "{\"role\": \"$2\", \"content\": \"$3\", \"channel\": \"${4:-telegram}\"}" | python3 -m json.tool
    ;;
  recent)
    curl -s "$BASE/conversation/recent?limit=${2:-50}&channel=${3}" | python3 -m json.tool
    ;;
  conversation-search)
    curl -s "$BASE/conversation/search?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${2}'))")&limit=${3:-20}" | python3 -m json.tool
    ;;
  entity-store)
    curl -s -X POST "$BASE/entity" -H "Content-Type: application/json" \
      -d "{\"name\": \"$2\", \"type\": \"$3\", \"details\": $4}" | python3 -m json.tool
    ;;
  entity-search)
    curl -s "$BASE/entity/search?q=$(python3 -c "import urllib.parse; print(urllib.parse.quote('${2:-}'))")&type=${3}" | python3 -m json.tool
    ;;
  *)
    echo "Usage: memory.sh {stats|health|recall|search|store|list|delete|log|recent|conversation-search|entity-store|entity-search}"
    echo ""
    echo "  stats                          - Show memory statistics"
    echo "  health                         - Check if API is running"
    echo "  recall <topic>                 - Recall memories + conversations about topic"
    echo "  search <query> [type] [limit]  - Search memories"
    echo "  store <name> <type> <content>  - Store a memory"
    echo "  list [type] [limit]            - List memories"
    echo "  delete <id>                    - Delete memory by ID"
    echo "  log <role> <content> [channel] - Log conversation message"
    echo "  recent [limit] [channel]       - Recent conversations"
    echo "  conversation-search <query>    - Search conversations"
    echo "  entity-store <name> <type> <json> - Store entity"
    echo "  entity-search [query] [type]   - Search entities"
    exit 1
    ;;
esac
