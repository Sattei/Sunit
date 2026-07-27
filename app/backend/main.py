from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from redis.exceptions import RedisError
from rq.exceptions import NoSuchJobError
from rq.job import Job

from backend.config import JOBS_DIR, QUEUE_NAME
from backend.queue import DEFAULT_JOB_TIMEOUT_SECONDS, redis_conn, relight_queue
from backend.tasks import run_auto_relight_job, run_relight_job


app = FastAPI(title="Sunit API")
AUTO_JOB_TIMEOUT_SECONDS = 1800
JOB_RESULT_TTL_SECONDS = 86400
JOB_FAILURE_TTL_SECONDS = 86400

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=JOBS_DIR), name="outputs")


def _save_upload(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    upload.file.seek(0)
    with destination.open("wb") as output_file:
        shutil.copyfileobj(upload.file, output_file)


def _status_value(job: Job) -> str:
    status = job.get_status(refresh=True)
    return str(getattr(status, "value", status))


def _tail(text: str | None, limit: int = 4000) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[-limit:]


@app.get("/")
def health() -> dict:
    return {
        "name": "Sunit API",
        "status": "ok",
        "queue": QUEUE_NAME,
        "endpoints": {
            "relight": "POST /api/relight for image + normal map",
            "relight_auto": "POST /api/relight-auto for image only",
        },
    }


@app.post("/api/relight")
def create_relight_job(
    image: UploadFile = File(...),
    normal: UploadFile = File(...),
    old_x: float = Form(-0.25),
    old_y: float = Form(-0.15),
    old_z: float = Form(1.0),
    new_x: float = Form(0.55),
    new_y: float = Form(-0.20),
    new_z: float = Form(0.80),
    person_strength: float = Form(0.75),
    background_strength: float = Form(0.18),
    ambient: float = Form(0.36),
    highlight: float = Form(0.12),
    save_debug: bool = Form(True),
) -> dict:
    job_id = uuid4().hex
    job_dir = JOBS_DIR / job_id

    image_path = job_dir / "input.png"
    normal_path = job_dir / "normal.png"
    output_path = job_dir / "output.png"

    _save_upload(image, image_path)
    _save_upload(normal, normal_path)

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
        relight_queue.enqueue(
            run_relight_job,
            payload,
            job_id=job_id,
            job_timeout=DEFAULT_JOB_TIMEOUT_SECONDS,
        )
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}") from exc

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/jobs/{job_id}",
    }


@app.post("/api/relight-auto")
def create_auto_relight_job(
    image: UploadFile = File(...),
    old_x: float = Form(-0.25),
    old_y: float = Form(-0.15),
    old_z: float = Form(1.0),
    new_x: float = Form(0.55),
    new_y: float = Form(-0.20),
    new_z: float = Form(0.80),
    person_strength: float = Form(0.65),
    background_strength: float = Form(0.0),
    ambient: float = Form(0.38),
    highlight: float = Form(0.08),
    save_debug: bool = Form(False),
) -> dict:
    job_id = uuid4().hex
    job_dir = JOBS_DIR / job_id

    image_path = job_dir / "input.png"
    output_path = job_dir / "output.png"

    _save_upload(image, image_path)

    payload = {
        "job_id": job_id,
        "image_path": str(image_path),
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
            run_auto_relight_job,
            payload,
            job_id=job_id,
            job_timeout=AUTO_JOB_TIMEOUT_SECONDS,
            result_ttl=JOB_RESULT_TTL_SECONDS,
            failure_ttl=JOB_FAILURE_TTL_SECONDS,
        )
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}") from exc

    return {
        "job_id": job.id,
        "status": "queued",
        "mode": "auto_normal",
        "status_url": f"/api/jobs/{job.id}",
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {exc}") from exc

    status = _status_value(job)
    response = {
        "job_id": job_id,
        "status": status,
    }

    if status == "finished":
        result = job.result or {}
        output_path_value = result.get("output_path")

        if output_path_value:
            try:
                relative_output = (
                    Path(output_path_value)
                    .resolve()
                    .relative_to(JOBS_DIR.resolve())
                )

                response["output_url"] = (
                    f"/outputs/{relative_output.as_posix()}"
                )

            except ValueError:
                response["output_url"] = (
                    f"/outputs/{job_id}/output/relighted.png"
                )
        else:
            response["output_url"] = (
                f"/outputs/{job_id}/output.png"
            )

        response["result"] = result
    elif status == "failed":
        response["error"] = _tail(job.exc_info) or "Job failed without error details."

    return response
