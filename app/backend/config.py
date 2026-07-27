from __future__ import annotations

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

STORAGE_DIR = APP_ROOT / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "sunit-relight")
INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "auto")

RELIGHT_SCRIPT = APP_ROOT / "scripts" / "relight_image.py"
DSINE_ROOT = Path(
    os.getenv(
        "DSINE_ROOT",
        PROJECT_ROOT / "external" / "DSINE",
    )
).expanduser().resolve()

DSINE_PYTHON_VALUE = os.getenv("DSINE_PYTHON")
DSINE_PYTHON = (
    Path(
        os.path.abspath(
            os.path.expanduser(DSINE_PYTHON_VALUE)
        )
    )
    if DSINE_PYTHON_VALUE
    else None
)

JOBS_DIR.mkdir(parents=True, exist_ok=True)
