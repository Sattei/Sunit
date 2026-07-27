#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT="${REPO_ROOT}/app"
RUNTIME_ROOT="${REPO_ROOT}/.runtime"
API_BASE_URL="${SUNIT_API_BASE_URL:-http://127.0.0.1:${SUNIT_API_PORT:-8000}}"
SMOKE_IMAGE_PATH="${SMOKE_IMAGE:-${APP_ROOT}/input/portrait.jpg}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-900}"

[[ -f "${SMOKE_IMAGE_PATH}" ]] || {
  echo "Smoke image not found: ${SMOKE_IMAGE_PATH}" >&2
  echo "Set SMOKE_IMAGE=/absolute/path/to/a/portrait.jpg and retry." >&2
  exit 1
}

mkdir -p "${RUNTIME_ROOT}/smoke"
curl --silent --show-error --fail "${API_BASE_URL}/health" >/dev/null

submit_json="$(
  curl --silent --show-error --fail \
    -X POST "${API_BASE_URL}/api/relight-auto" \
    -F "image=@${SMOKE_IMAGE_PATH}" \
    -F "new_x=0.55" \
    -F "new_y=-0.20" \
    -F "new_z=0.80" \
    -F "person_strength=0.65" \
    -F "ambient=0.38" \
    -F "highlight=0.08" \
    -F "preset=natural"
)"

job_id="$(
  SUBMIT_JSON="${submit_json}" "${APP_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

payload = json.loads(os.environ["SUBMIT_JSON"])
job_id = payload.get("job_id")
if not job_id:
    raise SystemExit(f"Submission did not return a job ID: {payload}")
print(job_id)
PY
)"

echo "Queued smoke job: ${job_id}"
deadline=$((SECONDS + SMOKE_TIMEOUT_SECONDS))
status=""
status_json=""

while (( SECONDS < deadline )); do
  status_json="$(curl --silent --show-error --fail "${API_BASE_URL}/api/jobs/${job_id}")"
  read -r status stage progress < <(
    STATUS_JSON="${status_json}" "${APP_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

payload = json.loads(os.environ["STATUS_JSON"])
print(payload.get("status", ""), payload.get("stage", ""), payload.get("progress", 0))
PY
  )
  echo "  ${stage:-unknown}: ${progress}% (${status:-unknown})"

  if [[ "${status}" == "finished" ]]; then
    break
  fi
  if [[ "${status}" == "failed" || "${status}" == "stopped" || "${status}" == "canceled" || "${status}" == "cancelled" ]]; then
    echo "Smoke job failed: ${status_json}" >&2
    exit 1
  fi
  sleep 2
done

[[ "${status}" == "finished" ]] || {
  echo "Smoke job timed out after ${SMOKE_TIMEOUT_SECONDS} seconds." >&2
  exit 1
}

result_json="$(curl --silent --show-error --fail "${API_BASE_URL}/api/jobs/${job_id}/result")"
output_url="$(
  RESULT_JSON="${result_json}" "${APP_ROOT}/.venv/bin/python" - <<'PY'
import json
import os

payload = json.loads(os.environ["RESULT_JSON"])
output_url = payload.get("output_url")
if not output_url:
    raise SystemExit(f"Result did not return an output URL: {payload}")
print(output_url)
PY
)"

output_file="${RUNTIME_ROOT}/smoke/${job_id}.png"
curl --silent --show-error --fail "${API_BASE_URL}${output_url}" -o "${output_file}"

"${APP_ROOT}/.venv/bin/python" - "${output_file}" <<'PY'
import sys
from pathlib import Path
from PIL import Image

path = Path(sys.argv[1])
with Image.open(path) as image:
    image.verify()
print(f"Valid output image: {path}")
PY

echo "Smoke test passed."
echo "Job ID: ${job_id}"
echo "Output: ${output_file}"
