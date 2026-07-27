from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.config import APP_ROOT, DSINE_ROOT, JOBS_DIR, STAGE5_SCRIPT
from src.pipeline.auto_relight import (
    PipelineError,
    RelightSettings,
    run_auto_relight,
)
from backend.queue import DEFAULT_JOB_TIMEOUT_SECONDS


STDOUT_TAIL_CHARS = 4000


def _tail(text: str, limit: int = STDOUT_TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _path_from_payload(payload: dict[str, Any], key: str) -> Path:
    try:
        return Path(payload[key])
    except KeyError as exc:
        raise ValueError(f"Missing required payload field: {key}") from exc


def _float_from_payload(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload[key])
    except KeyError as exc:
        raise ValueError(f"Missing required payload field: {key}") from exc


def run_relight_job(payload: dict) -> dict:
    """Run the existing Stage 5 relighting script in a worker subprocess."""
    image_path = _path_from_payload(payload, "image_path")
    normal_path = _path_from_payload(payload, "normal_path")
    output_path = _path_from_payload(payload, "output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(STAGE5_SCRIPT),
        "--image",
        str(image_path),
        "--normal",
        str(normal_path),
        "--output",
        str(output_path),
        "--old-light",
        str(_float_from_payload(payload, "old_x")),
        str(_float_from_payload(payload, "old_y")),
        str(_float_from_payload(payload, "old_z")),
        "--new-light",
        str(_float_from_payload(payload, "new_x")),
        str(_float_from_payload(payload, "new_y")),
        str(_float_from_payload(payload, "new_z")),
        "--person-strength",
        str(_float_from_payload(payload, "person_strength")),
        "--background-strength",
        str(_float_from_payload(payload, "background_strength")),
        "--ambient",
        str(_float_from_payload(payload, "ambient")),
        "--highlight",
        str(_float_from_payload(payload, "highlight")),
    ]

    if bool(payload.get("save_debug", False)):
        cmd.append("--save-debug")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(APP_ROOT),
            capture_output=True,
            text=True,
            timeout=DEFAULT_JOB_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Stage 5 relighting timed out after "
            f"{DEFAULT_JOB_TIMEOUT_SECONDS} seconds.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout tail:\n{_tail(exc.stdout or '')}\n"
            f"stderr tail:\n{_tail(exc.stderr or '')}"
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "Stage 5 relighting failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout tail:\n{_tail(completed.stdout)}\n"
            f"stderr tail:\n{_tail(completed.stderr)}"
        )

    return {
        "status": "finished",
        "output_path": str(output_path),
        "stdout_tail": _tail(completed.stdout),
    }


def run_auto_relight_job(payload: dict) -> dict:
    """
    Run the complete automatic Sunit pipeline:

        input image
            -> DSINE normal estimation
            -> BiRefNet soft matte
            -> canonical V8 relighting
    """
    image_path = _path_from_payload(payload, "image_path")

    job_id = str(
        payload.get("job_id")
        or image_path.parent.name
    )

    ambient = float(
        payload.get("ambient", 0.38)
    )

    settings = RelightSettings(
        old_light=(
            _float_from_payload(payload, "old_x"),
            _float_from_payload(payload, "old_y"),
            _float_from_payload(payload, "old_z"),
        ),
        new_light=(
            _float_from_payload(payload, "new_x"),
            _float_from_payload(payload, "new_y"),
            _float_from_payload(payload, "new_z"),
        ),
        ambient_old=ambient,
        ambient_new=ambient,
        strength=float(
            payload.get("person_strength", 0.65)
        ),
        boundary_relight=float(
            payload.get("boundary_relight", 0.30)
        ),
        background_strength=float(
            payload.get("background_strength", 0.0)
        ),
        shadow_relight=float(
            payload.get("shadow_relight", 0.45)
        ),
        albedo_floor=float(
            payload.get("albedo_floor", 0.28)
        ),
        specular_strength=float(
            payload.get("highlight", 0.08)
        ),
        specular_shininess=float(
            payload.get("specular_shininess", 72.0)
        ),
        max_darken_amount=float(
            payload.get("max_darken", 0.18)
        ),
        exposure=float(
            payload.get("exposure", 1.0)
        ),
        auto_old_light=bool(
            payload.get("auto_old_light", False)
        ),
        background_lock=True,
    )

    try:
        result = run_auto_relight(
            image_path=image_path,
            output_root=JOBS_DIR,
            dsine_root=DSINE_ROOT,
            device=str(payload.get("device", "auto")),
            job_id=job_id,
            save_debug=bool(
                payload.get("save_debug", False)
            ),
            settings=settings,
        )

    except PipelineError as error:
        raise RuntimeError(
            "Automatic Sunit pipeline failed.\n"
            f"Stage: {error.stage}\n"
            f"Details: {error}"
        ) from error

    return {
        "status": "done",
        "job_id": job_id,
        "image_path": result["input_path"],
        "normal_path": result["normal_path"],
        "matte_path": result["matte_path"],
        "relight_strength_path": result[
            "relight_strength_path"
        ],
        "output_path": result["output_path"],
    }
