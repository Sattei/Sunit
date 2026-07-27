#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="${REPO_ROOT}/app"
FRONTEND_ROOT="${APP_ROOT}/frontend"
VITE_BIN="${FRONTEND_ROOT}/node_modules/.bin/vite"
RUNTIME_ROOT="${REPO_ROOT}/.runtime"
LOG_ROOT="${RUNTIME_ROOT}/logs"
PYTHON_BIN="${APP_ROOT}/.venv/bin/python"
UVICORN_BIN="${APP_ROOT}/.venv/bin/uvicorn"
RQ_BIN="${APP_ROOT}/.venv/bin/rq"
DSINE_ROOT_VALUE="${DSINE_ROOT:-${REPO_ROOT}/external/DSINE}"
DSINE_CHECKPOINT="${DSINE_ROOT_VALUE}/projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt"
REDIS_URL_VALUE="${REDIS_URL:-redis://localhost:6379/0}"
QUEUE_NAME_VALUE="${QUEUE_NAME:-sunit-relight}"
API_PORT="${SUNIT_API_PORT:-8000}"
FRONTEND_PORT="${SUNIT_FRONTEND_PORT:-5173}"

mkdir -p "${LOG_ROOT}"

fail() {
  echo "Sunit startup failed: $*" >&2
  exit 1
}

require_executable() {
  [[ -x "$1" ]] || fail "required executable not found: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

pid_file_is_live() {
  local pid_file="$1"
  local pid=""
  local expected_start=""
  local actual_start=""

  [[ -f "${pid_file}" ]] || return 1
  read -r pid expected_start < "${pid_file}" || return 1
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${pid}/stat" ]] || return 1
  actual_start="$(awk '{print $22}' "/proc/${pid}/stat")"
  [[ "${actual_start}" == "${expected_start}" ]]
}

port_is_listening() {
  ss -H -ltn "sport = :$1" | grep -q .
}

show_port_owner() {
  local port="$1"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
}

record_pid() {
  local service="$1"
  local pid="$2"
  local start_time=""

  for _ in $(seq 1 20); do
    if [[ -r "/proc/${pid}/stat" ]]; then
      start_time="$(awk '{print $22}' "/proc/${pid}/stat")"
      break
    fi
    sleep 0.1
  done

  [[ -n "${start_time}" ]] || fail "${service} exited before its PID could be recorded"
  printf '%s %s\n' "${pid}" "${start_time}" > "${RUNTIME_ROOT}/${service}.pid"
}

for service in api worker frontend; do
  if pid_file_is_live "${RUNTIME_ROOT}/${service}.pid"; then
    fail "Sunit already has a running ${service} process. Run 'make dev-down' first."
  fi
  rm -f "${RUNTIME_ROOT}/${service}.pid"
done

require_executable "${PYTHON_BIN}"
require_executable "${UVICORN_BIN}"
require_executable "${RQ_BIN}"
require_command docker
require_command curl
require_command ss
require_command lsof
require_command setsid

docker compose version >/dev/null 2>&1 || fail "'docker compose' is unavailable"
[[ -d "${DSINE_ROOT_VALUE}" ]] || fail "DSINE repository not found: ${DSINE_ROOT_VALUE}"
[[ -f "${DSINE_CHECKPOINT}" ]] || fail "DSINE checkpoint not found: ${DSINE_CHECKPOINT}"
[[ -f "${FRONTEND_ROOT}/package.json" ]] || fail "frontend package not found"
[[ -x "${VITE_BIN}" ]] || fail "frontend dependencies are missing; run 'npm ci --prefix app/frontend'"

for port in "${API_PORT}" "${FRONTEND_PORT}"; do
  if port_is_listening "${port}"; then
    echo "Port ${port} is already occupied:" >&2
    show_port_owner "${port}" >&2
    fail "choose a free port with SUNIT_API_PORT or SUNIT_FRONTEND_PORT"
  fi
done

redis_was_running=false
if [[ "$(docker compose -f "${REPO_ROOT}/docker-compose.yml" ps --status running -q redis)" ]]; then
  redis_was_running=true
elif port_is_listening 6379; then
  echo "Port 6379 is already occupied:" >&2
  show_port_owner 6379 >&2
  fail "stop the unrelated Redis service or set a different REDIS_URL and Compose mapping"
fi

if [[ "${redis_was_running}" == false ]]; then
  docker compose -f "${REPO_ROOT}/docker-compose.yml" up -d redis
  touch "${RUNTIME_ROOT}/redis.started"
fi

redis_ready=false
for _ in $(seq 1 30); do
  if docker compose -f "${REPO_ROOT}/docker-compose.yml" exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    redis_ready=true
    break
  fi
  sleep 1
done
[[ "${redis_ready}" == true ]] || fail "Redis did not become healthy within 30 seconds"

COMMON_ENV=(
  "PYTHONPATH=${APP_ROOT}"
  "REDIS_URL=${REDIS_URL_VALUE}"
  "QUEUE_NAME=${QUEUE_NAME_VALUE}"
  "DSINE_ROOT=${DSINE_ROOT_VALUE}"
  "DSINE_PYTHON=${PYTHON_BIN}"
  "INFERENCE_DEVICE=${INFERENCE_DEVICE:-auto}"
)

(
  cd "${APP_ROOT}"
  nohup setsid env "${COMMON_ENV[@]}" \
    "${RQ_BIN}" worker "${QUEUE_NAME_VALUE}" --url "${REDIS_URL_VALUE}" \
    > "${LOG_ROOT}/worker.log" 2>&1 &
  record_pid worker "$!"
)

(
  cd "${APP_ROOT}"
  nohup setsid env "${COMMON_ENV[@]}" \
    "${UVICORN_BIN}" backend.main:app --host 127.0.0.1 --port "${API_PORT}" \
    > "${LOG_ROOT}/api.log" 2>&1 &
  record_pid api "$!"
)

(
  cd "${FRONTEND_ROOT}"
  nohup setsid env \
    "VITE_API_BASE_URL=http://localhost:${API_PORT}" \
    "${VITE_BIN}" --host 127.0.0.1 --port "${FRONTEND_PORT}" \
    > "${LOG_ROOT}/frontend.log" 2>&1 &
  record_pid frontend "$!"
)

api_ready=false
frontend_ready=false
for _ in $(seq 1 30); do
  if curl --silent --fail "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    api_ready=true
  fi
  if curl --silent --fail "http://127.0.0.1:${FRONTEND_PORT}/" >/dev/null; then
    frontend_ready=true
  fi
  if [[ "${api_ready}" == true && "${frontend_ready}" == true ]]; then
    break
  fi
  sleep 1
done

if [[ "${api_ready}" != true || "${frontend_ready}" != true ]]; then
  echo "A Sunit service did not become ready. Recent logs:" >&2
  tail -n 30 "${LOG_ROOT}/api.log" "${LOG_ROOT}/worker.log" "${LOG_ROOT}/frontend.log" >&2 || true
  "${REPO_ROOT}/scripts/dev_down.sh" >/dev/null 2>&1 || true
  fail "startup readiness check failed"
fi

echo "Sunit is running."
echo "Frontend: http://localhost:${FRONTEND_PORT}"
echo "API:      http://localhost:${API_PORT}"
echo "Logs:     ${LOG_ROOT}"
echo "Stop:     make dev-down"
