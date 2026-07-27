from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.pipeline.auto_relight import (
    PipelineError,
    RelightSettings,
    prepare_job_directories,
    run_auto_relight,
)


class FakeDSINE:
    def estimate(self, input_image: Path, destination: Path) -> Path:
        with Image.open(input_image) as source:
            width, height = source.size

        normal = np.zeros((height, width, 3), dtype=np.uint8)
        normal[:, :, 0:2] = 128
        normal[:, :, 2] = 255
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normal, mode="RGB").save(destination)
        return destination


class FakeMatte:
    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def predict(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        return np.full((height, width), self.value, dtype=np.float32)


def make_image(path: Path, size: tuple[int, int] = (160, 160)) -> Path:
    image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)
    return path


def test_relight_settings_defaults() -> None:
    settings = RelightSettings()

    assert settings.strength == 0.65
    assert settings.background_lock is True
    assert settings.background_strength == 0.0


def test_missing_input_preserves_validation_stage(tmp_path: Path) -> None:
    with pytest.raises(PipelineError) as raised:
        run_auto_relight(
            image_path=tmp_path / "missing.jpg",
            output_root=tmp_path / "jobs",
            dsine_root=tmp_path / "dsine",
        )

    assert raised.value.stage == "validation"


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "portrait.gif"
    source.write_bytes(b"not-an-image")

    with pytest.raises(PipelineError) as raised:
        run_auto_relight(
            image_path=source,
            output_root=tmp_path / "jobs",
            dsine_root=tmp_path / "dsine",
        )

    assert raised.value.stage == "validation"
    assert "Unsupported image type" in raised.value.message


def test_existing_job_requires_explicit_overwrite(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    prepare_job_directories(jobs, "repeat")

    with pytest.raises(PipelineError) as raised:
        prepare_job_directories(jobs, "repeat")

    assert raised.value.stage == "input_preparation"

    directories = prepare_job_directories(jobs, "repeat", overwrite=True)
    assert directories["job"].is_dir()


def test_precreated_upload_directory_is_safe(tmp_path: Path) -> None:
    upload = tmp_path / "jobs" / "api-job" / "upload"
    upload.mkdir(parents=True)
    (upload / "original.jpg").write_bytes(b"upload")

    directories = prepare_job_directories(
        tmp_path / "jobs",
        "api-job",
        accept_precreated_upload=True,
    )

    assert directories["input"].is_dir()
    assert (upload / "original.jpg").exists()


def test_pipeline_reports_stages_and_resizes_processing_input(
    tmp_path: Path,
) -> None:
    source = make_image(tmp_path / "portrait.png", (200, 200))
    progress: list[tuple[str, int]] = []

    result = run_auto_relight(
        image_path=source,
        output_root=tmp_path / "jobs",
        dsine_root=tmp_path / "dsine",
        job_id="mocked",
        matte_model=FakeMatte(),
        dsine_adapter=FakeDSINE(),
        progress_callback=lambda stage, value: progress.append((stage, value)),
        max_processing_side=128,
    )

    assert progress == [
        ("validating", 5),
        ("preparing_input", 10),
        ("estimating_normals", 20),
        ("generating_matte", 50),
        ("relighting", 75),
        ("saving_output", 95),
        ("completed", 100),
    ]
    assert Path(result["output_path"]).is_file()
    assert Image.open(result["output_path"]).size == (128, 128)
    assert Path(result["original_input_path"]).is_file()


def test_empty_matte_is_reported_as_matte_generation_failure(
    tmp_path: Path,
) -> None:
    source = make_image(tmp_path / "portrait.png")

    with pytest.raises(PipelineError) as raised:
        run_auto_relight(
            image_path=source,
            output_root=tmp_path / "jobs",
            dsine_root=tmp_path / "dsine",
            job_id="no-subject",
            matte_model=FakeMatte(0.0),
            dsine_adapter=FakeDSINE(),
        )

    assert raised.value.stage == "matte_generation"
    assert raised.value.message == "No clear foreground subject was detected."
