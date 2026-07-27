#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${REPO_ROOT}/.runtime"

stop_service() {
  local service="$1"
  local pid_file="${RUNTIME_ROOT}/${service}.pid"
  local pid=""
  local expected_start=""
  local actual_start=""

  [[ -f "${pid_file}" ]] || return 0
  read -r pid expected_start < "${pid_file}" || {
    echo "Ignoring malformed ${service} PID file." >&2
    rm -f "${pid_file}"
    return 0
  }

  if [[ ! "${pid}" =~ ^[0-9]+$ || ! -r "/proc/${pid}/stat" ]]; then
    rm -f "${pid_file}"
    return 0
  fi

  actual_start="$(awk '{print $22}' "/proc/${pid}/stat")"
  if [[ "${actual_start}" != "${expected_start}" ]]; then
    echo "Not stopping PID ${pid}: stale ${service} PID file." >&2
    rm -f "${pid_file}"
    return 0
  fi

  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true

  for _ in $(seq 1 50); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done

  if kill -0 "${pid}" 2>/dev/null; then
    echo "${service} did not stop gracefully; forcing its Sunit process group to exit." >&2
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi

  rm -f "${pid_file}"
  echo "Stopped ${service}."
}

stop_service frontend
stop_service api
stop_service worker

if [[ -f "${RUNTIME_ROOT}/redis.started" ]]; then
  docker compose -f "${REPO_ROOT}/docker-compose.yml" stop redis >/dev/null
  rm -f "${RUNTIME_ROOT}/redis.started"
  echo "Stopped Sunit Redis."
fi

echo "Sunit is stopped. Logs were preserved under ${RUNTIME_ROOT}/logs/."
