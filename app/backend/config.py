from __future__ import annotations

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]

STORAGE_DIR = APP_ROOT / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_NAME = os.getenv("QUEUE_NAME", "sunit-relight")

STAGE5_SCRIPT = APP_ROOT / "src" / "relighting" / "stage5_depth_face_relight.py"
DSINE_ADAPTER = APP_ROOT / "src" / "normal_estimation" / "dsine_adapter.py"
DSINE_ROOT = APP_ROOT.parent / "external" / "DSINE"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
