#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="${REPO_ROOT}/app"
API_PORT="${SUNIT_API_PORT:-8000}"
FRONTEND_PORT="${SUNIT_FRONTEND_PORT:-5173}"
REDIS_URL_VALUE="${REDIS_URL:-redis://localhost:6379/0}"
QUEUE_NAME_VALUE="${QUEUE_NAME:-sunit-relight}"

docker compose -f "${REPO_ROOT}/docker-compose.yml" exec -T redis redis-cli ping \
  | grep -q PONG
echo "Redis:   ok"

health_json="$(curl --silent --show-error --fail "http://127.0.0.1:${API_PORT}/health")"
HEALTH_JSON="${health_json}" "${APP_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

payload = json.loads(os.environ["HEALTH_JSON"])
if payload.get("status") != "ok" or payload.get("redis") != "ok":
    raise SystemExit(f"API health is not ok: {payload}")
print(
    "API:     ok "
    f"(GPU={payload.get('gpu_available')}, "
    f"DSINE={payload.get('dsine_checkpoint_available')})"
)
PY

curl --silent --show-error --fail "http://127.0.0.1:${FRONTEND_PORT}/" >/dev/null
echo "Frontend: ok"

workers="$(
  PYTHONPATH="${APP_ROOT}" "${APP_ROOT}/.venv/bin/rq" info \
    --url "${REDIS_URL_VALUE}" --only-workers
)"
if ! grep -q "${QUEUE_NAME_VALUE}" <<< "${workers}"; then
  echo "No RQ worker is registered for ${QUEUE_NAME_VALUE}." >&2
  exit 1
fi
echo "Worker:   ok (${QUEUE_NAME_VALUE})"

echo "All Sunit health checks passed."
