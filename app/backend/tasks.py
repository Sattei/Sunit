from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from rq import get_current_job

from backend.config import (
    APP_ROOT,
    DSINE_PYTHON,
    DSINE_ROOT,
    INFERENCE_DEVICE,
    JOBS_DIR,
    RELIGHT_SCRIPT,
)
from backend.queue import DEFAULT_JOB_TIMEOUT_SECONDS
from src.masking.birefnet_matte import BiRefNetMatte
from src.pipeline.auto_relight import (
    PipelineError,
    RelightSettings,
    run_auto_relight,
)


LOGGER = logging.getLogger(__name__)
STDOUT_TAIL_CHARS = 4000
PUBLIC_STAGE_ERRORS = {
    "validation": "The uploaded image is invalid.",
    "input_preparation": "The image could not be prepared.",
    "normal_estimation": "Surface-normal estimation failed.",
    "matte_generation": "We could not isolate the portrait subject.",
    "relighting": "The relighting operation failed.",
    "output_saving": "The relighted image could not be saved.",
}

_BIREFNET_MODEL: BiRefNetMatte | None = None


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


def _update_job_progress(stage: str, progress: int) -> None:
    job = get_current_job()

    if job is None:
        return

    job.meta["stage"] = stage
    job.meta["progress"] = progress
    job.save_meta()


def _record_public_failure(error: PipelineError) -> None:
    job = get_current_job()

    if job is None:
        return

    message = PUBLIC_STAGE_ERRORS.get(
        error.stage,
        "Image processing failed.",
    )

    if (
        error.stage == "matte_generation"
        and "No clear foreground subject" in error.message
    ):
        message = (
            "We could not detect a clear portrait subject. "
            "Try a closer or clearer image."
        )

    job.meta["stage"] = error.stage
    job.meta["progress"] = job.meta.get("progress", 0)
    job.meta["public_error"] = {
        "stage": error.stage,
        "message": message,
    }
    job.save_meta()


def _record_unexpected_failure() -> None:
    job = get_current_job()

    if job is None:
        return

    job.meta["stage"] = str(job.meta.get("stage") or "processing")
    job.meta["progress"] = job.meta.get("progress", 0)
    job.meta["public_error"] = {
        "stage": job.meta["stage"],
        "message": "Image processing failed. Please try again.",
    }
    job.save_meta()


def get_birefnet_model() -> BiRefNetMatte:
    global _BIREFNET_MODEL

    if _BIREFNET_MODEL is None:
        _BIREFNET_MODEL = BiRefNetMatte(device=INFERENCE_DEVICE)
        _BIREFNET_MODEL.load()

    return _BIREFNET_MODEL


def run_relight_job(payload: dict) -> dict:
    """Run the canonical engine CLI for the advanced image + normal workflow."""
    image_path = _path_from_payload(payload, "image_path")
    normal_path = _path_from_payload(payload, "normal_path")
    output_path = _path_from_payload(payload, "output_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RELIGHT_SCRIPT),
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
    _update_job_progress("queued", 0)

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
            dsine_python=DSINE_PYTHON,
            device=str(payload.get("device", INFERENCE_DEVICE)),
            job_id=job_id,
            save_debug=bool(
                payload.get("save_debug", False)
            ),
            settings=settings,
            matte_model=get_birefnet_model(),
            progress_callback=_update_job_progress,
            accept_precreated_upload=True,
        )

    except PipelineError as error:
        _record_public_failure(error)
        LOGGER.exception(
            "Automatic Sunit pipeline failed at stage %s",
            error.stage,
        )
        raise
    except Exception:
        _record_unexpected_failure()
        LOGGER.exception("Automatic Sunit pipeline failed unexpectedly")
        raise

    _update_job_progress("completed", 100)

    return {
        "status": "finished",
        "job_id": job_id,
        "output_path": result["output_path"],
    }
