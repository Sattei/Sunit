from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from backend.config import (
    DSINE_PYTHON,
    DSINE_ROOT,
    JOBS_DIR,
    QUEUE_NAME,
)
from backend.queue import DEFAULT_JOB_TIMEOUT_SECONDS, redis_conn, relight_queue
from backend.tasks import run_auto_relight_job, run_relight_job
from src.pipeline.auto_relight import PipelineError, validate_image


LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Sunit API",
    description="Asynchronous single-image portrait relighting.",
    version="1.0.0",
)

AUTO_JOB_TIMEOUT_SECONDS = 1800
JOB_RESULT_TTL_SECONDS = 86400
JOB_FAILURE_TTL_SECONDS = 86400
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TERMINAL_FAILURE_STATUSES = {"failed", "stopped", "canceled", "cancelled"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=JOBS_DIR), name="outputs")

LOGGER.info(
    "Sunit inference configuration: DSINE root=%s, DSINE Python=%s",
    DSINE_ROOT,
    DSINE_PYTHON or "automatic",
)


def _safe_upload_suffix(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()

    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Use JPG, JPEG, PNG, or WEBP.",
        )

    return suffix


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    try:
        with destination.open("wb") as output_file:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)

                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Image is too large. Maximum upload size is 15 MB.",
                    )

                output_file.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="The uploaded image is empty.",
        )

    return total_bytes


def _validate_saved_upload(path: Path) -> None:
    try:
        validate_image(path)
    except PipelineError as error:
        raise HTTPException(
            status_code=422,
            detail=error.message,
        ) from error


def _validate_light_vector(x: float, y: float, z: float) -> None:
    if (x * x + y * y + z * z) < 1e-8:
        raise HTTPException(
            status_code=422,
            detail="Light direction cannot be zero.",
        )


def _status_value(job: Job) -> str:
    status = job.get_status(refresh=True)
    return str(getattr(status, "value", status))


def _job_progress(job: Job, status: str) -> tuple[str, int]:
    if status == "finished":
        return "completed", 100

    stage = str(job.meta.get("stage") or status)
    progress = int(job.meta.get("progress") or 0)
    return stage, max(0, min(100, progress))


def _output_url_for_result(
    job_id: str,
    result: dict[str, Any] | None,
) -> str:
    output_path_value = (result or {}).get("output_path")
    default_url = f"/outputs/{job_id}/output/relighted.png"

    if not output_path_value:
        return default_url

    try:
        relative_output = (
            Path(str(output_path_value))
            .resolve()
            .relative_to((JOBS_DIR / job_id).resolve())
        )
    except (OSError, ValueError):
        return default_url

    return f"/outputs/{job_id}/{relative_output.as_posix()}"


def _public_failure(job: Job) -> dict[str, str]:
    public_error = job.meta.get("public_error")

    if isinstance(public_error, dict):
        return {
            "stage": str(public_error.get("stage") or "processing"),
            "message": str(
                public_error.get("message")
                or "Image processing failed."
            ),
        }

    return {
        "stage": str(job.meta.get("stage") or "processing"),
        "message": "Image processing failed. Please try again.",
    }


def _fetch_job(job_id: str) -> Job:
    try:
        return Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error
    except RedisError as error:
        LOGGER.exception("Redis failed while fetching job %s", job_id)
        raise HTTPException(
            status_code=503,
            detail="The job service is temporarily unavailable.",
        ) from error


def _job_response(job: Job) -> dict[str, Any]:
    status = _status_value(job)
    stage, progress = _job_progress(job, status)
    response: dict[str, Any] = {
        "job_id": job.id,
        "status": status,
        "stage": stage,
        "progress": progress,
    }

    if status == "finished":
        response["output_url"] = _output_url_for_result(
            job.id,
            job.result if isinstance(job.result, dict) else None,
        )
    elif status in TERMINAL_FAILURE_STATUSES:
        response["error"] = _public_failure(job)

    return response


def _queue_unavailable(error: RedisError, job_id: str) -> HTTPException:
    LOGGER.exception("Redis unavailable while enqueueing job %s", job_id)
    return HTTPException(
        status_code=503,
        detail="The relighting queue is temporarily unavailable.",
    )


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "Sunit API",
        "description": "Geometry-aware single-image portrait relighting.",
        "primary_endpoint": "POST /api/relight-auto",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        redis_ok = bool(redis_conn.ping())
    except RedisError:
        redis_ok = False

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "ok" if redis_ok else "unavailable",
        "queue": QUEUE_NAME,
        "gpu_available": torch.cuda.is_available(),
        "birefnet_model": "ZhengPeng7/BiRefNet_lite-matting",
        "dsine_root_available": DSINE_ROOT.is_dir(),
        "dsine_checkpoint_available": (
            DSINE_ROOT
            / "projects"
            / "dsine"
            / "checkpoints"
            / "exp001_cvpr2024"
            / "dsine.pt"
        ).is_file(),
        "dsine_python": (
            DSINE_PYTHON.name
            if DSINE_PYTHON is not None
            else "automatic"
        ),
    }


@app.post("/api/relight")
async def create_relight_job(
    image: UploadFile = File(...),
    normal: UploadFile = File(...),
    old_x: float = Form(-0.25, ge=-1.0, le=1.0),
    old_y: float = Form(-0.15, ge=-1.0, le=1.0),
    old_z: float = Form(1.0, ge=-1.0, le=1.0),
    new_x: float = Form(0.55, ge=-1.0, le=1.0),
    new_y: float = Form(-0.20, ge=-1.0, le=1.0),
    new_z: float = Form(0.80, ge=-1.0, le=1.0),
    person_strength: float = Form(0.75, ge=0.0, le=1.0),
    background_strength: float = Form(0.18, ge=0.0, le=0.25),
    ambient: float = Form(0.36, ge=0.05, le=0.80),
    highlight: float = Form(0.12, ge=0.0, le=0.30),
    save_debug: bool = Form(False),
) -> dict[str, Any]:
    _validate_light_vector(old_x, old_y, old_z)
    _validate_light_vector(new_x, new_y, new_z)

    job_id = uuid4().hex
    job_dir = JOBS_DIR / job_id
    image_path = job_dir / "manual" / f"image{_safe_upload_suffix(image)}"
    normal_path = job_dir / "manual" / f"normal{_safe_upload_suffix(normal)}"
    output_path = job_dir / "manual" / "relighted.png"

    try:
        await _save_upload(image, image_path)
        await _save_upload(normal, normal_path)
        _validate_saved_upload(image_path)
        _validate_saved_upload(normal_path)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    payload = {
        "job_id": job_id,
        "image_path": str(image_path),
        "normal_path": str(normal_path),
        "output_path": str(output_path),
        "old_x": old_x,
        "old_y": old_y,
        "old_z": old_z,
        "new_x": new_x,
        "new_y": new_y,
        "new_z": new_z,
        "person_strength": person_strength,
        "background_strength": background_strength,
        "ambient": ambient,
        "highlight": highlight,
        "save_debug": save_debug,
    }

    try:
        job = relight_queue.enqueue(
            run_relight_job,
            payload,
            job_id=job_id,
            job_timeout=DEFAULT_JOB_TIMEOUT_SECONDS,
            result_ttl=JOB_RESULT_TTL_SECONDS,
            failure_ttl=JOB_FAILURE_TTL_SECONDS,
        )
    except RedisError as error:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise _queue_unavailable(error, job_id) from error

    return {
        "job_id": job.id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "status_url": f"/api/jobs/{job.id}",
        "result_url": f"/api/jobs/{job.id}/result",
    }


@app.post("/api/relight-auto")
async def create_auto_relight_job(
    image: UploadFile = File(...),
    new_x: float = Form(0.55, ge=-1.0, le=1.0),
    new_y: float = Form(-0.20, ge=-1.0, le=1.0),
    new_z: float = Form(0.80, ge=-1.0, le=1.0),
    person_strength: float = Form(0.65, ge=0.0, le=1.0),
    background_strength: float = Form(0.0, ge=0.0, le=0.25),
    ambient: float = Form(0.38, ge=0.05, le=0.80),
    highlight: float = Form(0.08, ge=0.0, le=0.30),
    boundary_relight: float = Form(0.30, ge=0.0, le=0.70),
    shadow_relight: float = Form(0.45, ge=0.0, le=1.0),
    save_debug: bool = Form(False),
    preset: str = Form("natural", pattern=r"^(natural|soft_portrait|dramatic|custom)$"),
) -> dict[str, Any]:
    _validate_light_vector(new_x, new_y, new_z)

    job_id = uuid4().hex
    job_dir = JOBS_DIR / job_id
    suffix = _safe_upload_suffix(image)
    image_path = job_dir / "upload" / f"original{suffix}"

    try:
        await _save_upload(image, image_path)
        _validate_saved_upload(image_path)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    payload = {
        "job_id": job_id,
        "image_path": str(image_path),
        "old_x": -0.25,
        "old_y": -0.15,
        "old_z": 1.0,
        "new_x": new_x,
        "new_y": new_y,
        "new_z": new_z,
        "person_strength": person_strength,
        "background_strength": background_strength,
        "ambient": ambient,
        "highlight": highlight,
        "boundary_relight": boundary_relight,
        "shadow_relight": shadow_relight,
        "save_debug": save_debug,
        "preset": preset,
    }

    try:
        job = relight_queue.enqueue(
            run_auto_relight_job,
            payload,
            job_id=job_id,
            job_timeout=AUTO_JOB_TIMEOUT_SECONDS,
            result_ttl=JOB_RESULT_TTL_SECONDS,
            failure_ttl=JOB_FAILURE_TTL_SECONDS,
        )
    except RedisError as error:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise _queue_unavailable(error, job_id) from error

    return {
        "job_id": job.id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "status_url": f"/api/jobs/{job.id}",
        "result_url": f"/api/jobs/{job.id}/result",
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return _job_response(_fetch_job(job_id))


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str) -> JSONResponse:
    job = _fetch_job(job_id)
    response = _job_response(job)
    status = response["status"]

    if status == "finished":
        return JSONResponse(status_code=200, content=response)

    if status in TERMINAL_FAILURE_STATUSES:
        return JSONResponse(status_code=500, content=response)

    return JSONResponse(status_code=202, content=response)
