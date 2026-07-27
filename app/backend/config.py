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

STAGE5_SCRIPT = APP_ROOT / "src" / "relighting" / "stage5_depth_face_relight.py"
DSINE_ADAPTER = APP_ROOT / "src" / "normal_estimation" / "dsine_adapter.py"
DSINE_ROOT = Path(
    os.getenv(
        "DSINE_ROOT",
        PROJECT_ROOT / "external" / "DSINE",
    )
).expanduser().resolve()

DSINE_PYTHON_VALUE = os.getenv("DSINE_PYTHON")
DSINE_PYTHON = (
    Path(DSINE_PYTHON_VALUE).expanduser().resolve()
    if DSINE_PYTHON_VALUE
    else None
)

JOBS_DIR.mkdir(parents=True, exist_ok=True)
