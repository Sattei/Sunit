from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import shutil
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL import UnidentifiedImageError


APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from src.masking.birefnet_matte import (
    BiRefNetMatte,
    build_relighting_masks,
    save_grayscale,
)
from src.normal_estimation.dsine_adapter import DSINEAdapter
from src.relighting.engine import (
    load_normal_map,
    load_rgb,
    relight_person_only,
    save_debug_maps,
    save_rgb,
)

LOGGER = logging.getLogger(__name__)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGE_SIDE = 128
MAX_IMAGE_SIDE = 4096
MAX_PROCESSING_SIDE = 1536
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

ProgressCallback = Callable[[str, int], None]


class PipelineError(RuntimeError):
    """Raised when one stage of the Sunit pipeline fails."""

    def __init__(
        self,
        stage: str,
        message: str,
    ) -> None:
        self.stage = stage
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class RelightSettings:
    old_light: tuple[float, float, float] = (-0.25, -0.15, 1.0)
    new_light: tuple[float, float, float] = (0.55, -0.20, 0.80)

    ambient_old: float = 0.45
    ambient_new: float = 0.38

    strength: float = 0.65
    boundary_relight: float = 0.30
    background_strength: float = 0.0
    shadow_relight: float = 0.45

    albedo_floor: float = 0.28

    specular_strength: float = 0.08
    specular_shininess: float = 72.0

    max_darken_amount: float = 0.18
    exposure: float = 1.0

    auto_old_light: bool = False
    background_lock: bool = True


def choose_dsine_python(
    dsine_root: Path,
    requested_python: Path | None,
) -> Path:
    if requested_python is not None:
        return requested_python.expanduser().resolve()

    dsine_venv_python = (
        dsine_root
        / ".venv"
        / "bin"
        / "python"
    )

    if (
        dsine_venv_python.is_file()
        and os.access(dsine_venv_python, os.X_OK)
    ):
        return dsine_venv_python.resolve()

    return Path(sys.executable).resolve()


def report_progress(
    callback: ProgressCallback | None,
    stage: str,
    progress: int,
) -> None:
    if callback is not None:
        callback(stage, progress)


def cleanup_inference_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_image(image_path: Path) -> tuple[int, int]:
    if not image_path.exists():
        raise PipelineError(
            "validation",
            f"Input image was not found: {image_path}",
        )

    if not image_path.is_file():
        raise PipelineError(
            "validation",
            f"Input path is not a file: {image_path}",
        )

    if image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise PipelineError(
            "validation",
            "Unsupported image type. Use JPG, JPEG, PNG, or WEBP.",
        )

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            width, height = image.size
            image.convert("RGB").load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise PipelineError(
            "validation",
            "The input file is not a decodable image.",
        ) from error

    if min(width, height) < MIN_IMAGE_SIDE:
        raise PipelineError(
            "validation",
            f"Image sides must be at least {MIN_IMAGE_SIDE} pixels.",
        )

    if max(width, height) > MAX_IMAGE_SIDE:
        raise PipelineError(
            "validation",
            f"Image sides must not exceed {MAX_IMAGE_SIDE} pixels.",
        )

    return width, height


def _validate_job_id(job_id: str) -> None:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise PipelineError(
            "input_preparation",
            "Job ID may contain only letters, numbers, underscores, and hyphens.",
        )


def prepare_job_directories(
    output_root: Path,
    job_id: str,
    *,
    overwrite: bool = False,
    accept_precreated_upload: bool = False,
) -> dict[str, Path]:
    _validate_job_id(job_id)

    job_directory = output_root / job_id

    if job_directory.exists():
        existing_names = {
            path.name
            for path in job_directory.iterdir()
        }
        upload_only = (
            accept_precreated_upload
            and existing_names
            and existing_names <= {"upload"}
        )

        if overwrite:
            shutil.rmtree(job_directory)
        elif not upload_only:
            raise PipelineError(
                "input_preparation",
                (
                    f"Job '{job_id}' already exists. "
                    "Use overwrite=True (or --overwrite) to replace it."
                ),
            )

    directories = {
        "job": job_directory,
        "input": job_directory / "input",
        "intermediate": job_directory / "intermediate",
        "output": job_directory / "output",
        "debug": job_directory / "debug",
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


def run_auto_relight(
    image_path: Path,
    output_root: Path,
    dsine_root: Path,
    dsine_python: Path | None = None,
    device: str = "auto",
    job_id: str | None = None,
    save_debug: bool = False,
    settings: RelightSettings | None = None,
    matte_model: BiRefNetMatte | None = None,
    dsine_adapter: DSINEAdapter | None = None,
    progress_callback: ProgressCallback | None = None,
    overwrite: bool = False,
    accept_precreated_upload: bool = False,
    max_processing_side: int = MAX_PROCESSING_SIDE,
) -> dict[str, Any]:
    image_path = image_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    dsine_root = dsine_root.expanduser().resolve()

    report_progress(progress_callback, "validating", 5)
    width, height = validate_image(image_path)

    settings = settings or RelightSettings()
    resolved_job_id = job_id or uuid.uuid4().hex[:12]

    directories = prepare_job_directories(
        output_root=output_root,
        job_id=resolved_job_id,
        overwrite=overwrite,
        accept_precreated_upload=accept_precreated_upload,
    )

    input_suffix = image_path.suffix.lower() or ".png"

    original_input = (
        directories["input"]
        / f"original{input_suffix}"
    )
    processing_input = original_input

    normal_path = (
        directories["intermediate"]
        / "normal.png"
    )

    matte_path = (
        directories["intermediate"]
        / "person_alpha.png"
    )

    relight_strength_path = (
        directories["intermediate"]
        / "relight_strength.png"
    )

    output_path = (
        directories["output"]
        / "relighted.png"
    )

    report_progress(progress_callback, "preparing_input", 10)

    try:
        shutil.copy2(image_path, original_input)

        if max(width, height) > max_processing_side:
            scale = max_processing_side / float(max(width, height))
            resized_size = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            processing_input = directories["input"] / "processed.png"

            with Image.open(original_input) as image:
                image.convert("RGB").resize(
                    resized_size,
                    Image.Resampling.LANCZOS,
                ).save(processing_input)
    except Exception as error:
        raise PipelineError(
            "input_preparation",
            "The input image could not be prepared for processing.",
        ) from error

    if dsine_adapter is None:
        resolved_dsine_python = choose_dsine_python(
            dsine_root=dsine_root,
            requested_python=dsine_python,
        )

        dsine_adapter = DSINEAdapter(
            dsine_root=dsine_root,
            python_executable=resolved_dsine_python,
        )

    print("\n[1/3] Estimating surface normals with DSINE...")
    report_progress(progress_callback, "estimating_normals", 20)

    try:
        dsine_adapter.estimate(
            input_image=processing_input,
            destination=normal_path,
        )
    except Exception as error:
        LOGGER.exception("DSINE normal estimation failed")
        raise PipelineError(
            "normal_estimation",
            "Surface-normal estimation failed.",
        ) from error

    owns_matte_model = matte_model is None

    if matte_model is None:
        matte_model = BiRefNetMatte(
            device=device,
        )

    print("\n[2/3] Generating the BiRefNet soft matte...")
    report_progress(progress_callback, "generating_matte", 50)

    try:
        with Image.open(processing_input) as image:
            alpha = matte_model.predict(
                image.convert("RGB")
            )

        foreground_fraction = float((alpha > 0.5).mean())
        mean_alpha = float(alpha.mean())

        if foreground_fraction < 0.01:
            raise PipelineError(
                "matte_generation",
                "No clear foreground subject was detected.",
            )

        masks = build_relighting_masks(alpha)

        save_grayscale(
            masks["alpha"],
            matte_path,
        )

        save_grayscale(
            masks["relight_strength"],
            relight_strength_path,
        )

        LOGGER.info(
            "BiRefNet matte statistics: foreground_fraction=%.4f mean_alpha=%.4f",
            foreground_fraction,
            mean_alpha,
        )

    except PipelineError:
        raise
    except Exception as error:
        LOGGER.exception("BiRefNet matte generation failed")
        raise PipelineError(
            "matte_generation",
            "Portrait matte generation failed.",
        ) from error

    finally:
        if owns_matte_model:
            matte_model.unload()
        cleanup_inference_memory()

    print("\n[3/3] Running the canonical V8 relighting engine...")
    report_progress(progress_callback, "relighting", 75)

    try:
        image_srgb = load_rgb(
            str(processing_input)
        )

        height, width = image_srgb.shape[:2]

        normal = load_normal_map(
            path=str(normal_path),
            target_hw=(height, width),
            flip_x=False,
            flip_y=False,
            flip_z=False,
        )

        output, debug_maps = relight_person_only(
            image_srgb=image_srgb,
            normal=normal,
            person_mask_path=str(matte_path),
            old_light=np.asarray(
                settings.old_light,
                dtype=np.float32,
            ),
            new_light=np.asarray(
                settings.new_light,
                dtype=np.float32,
            ),
            ambient_old=settings.ambient_old,
            ambient_new=settings.ambient_new,
            strength=settings.strength,
            boundary_relight=settings.boundary_relight,
            background_strength=settings.background_strength,
            shadow_relight=settings.shadow_relight,
            albedo_floor=settings.albedo_floor,
            specular_strength=settings.specular_strength,
            specular_shininess=settings.specular_shininess,
            max_darken_amount=settings.max_darken_amount,
            exposure=settings.exposure,
            auto_old_light_enabled=settings.auto_old_light,
            background_lock=settings.background_lock,
        )

    except Exception as error:
        LOGGER.exception("Canonical relighting failed")
        raise PipelineError(
            "relighting",
            "The relighting operation failed.",
        ) from error
    finally:
        cleanup_inference_memory()

    report_progress(progress_callback, "saving_output", 95)

    try:
        save_rgb(str(output_path), output)

        if save_debug:
            save_debug_maps(
                str(directories["debug"] / "relighted.png"),
                debug_maps,
            )
    except Exception as error:
        LOGGER.exception("Saving relighted output failed")
        raise PipelineError(
            "output_saving",
            "The relighted image could not be saved.",
        ) from error

    result = {
        "status": "completed",
        "job_id": resolved_job_id,
        "job_directory": str(directories["job"]),
        "input_path": str(processing_input),
        "original_input_path": str(original_input),
        "normal_path": str(normal_path),
        "matte_path": str(matte_path),
        "relight_strength_path": str(relight_strength_path),
        "output_path": str(output_path),
    }

    print("\nSunit automatic relighting completed.")
    print(f"Final output: {output_path}")
    report_progress(progress_callback, "completed", 100)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Sunit pipeline: "
            "DSINE normals, BiRefNet matte, and V8 relighting."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input portrait image.",
    )

    parser.add_argument(
        "--output-root",
        default=str(APP_ROOT / "output" / "jobs"),
        help="Directory in which job folders are created.",
    )

    parser.add_argument(
        "--dsine-root",
        default=str(PROJECT_ROOT / "external" / "DSINE"),
        help="Path to the DSINE repository.",
    )

    parser.add_argument(
        "--dsine-python",
        default=None,
        help="Optional Python executable used for DSINE.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used by BiRefNet.",
    )

    parser.add_argument(
        "--job-id",
        default=None,
        help="Optional fixed job identifier.",
    )

    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save V8 relighting debug maps.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated job directory with the same ID.",
    )

    parser.add_argument(
        "--old-light",
        nargs=3,
        type=float,
        default=(-0.25, -0.15, 1.0),
    )

    parser.add_argument(
        "--new-light",
        nargs=3,
        type=float,
        default=(0.55, -0.20, 0.80),
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--ambient-old",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--ambient-new",
        type=float,
        default=0.38,
    )

    parser.add_argument(
        "--boundary-relight",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--shadow-relight",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--highlight",
        type=float,
        default=0.08,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = RelightSettings(
        old_light=tuple(args.old_light),
        new_light=tuple(args.new_light),
        ambient_old=args.ambient_old,
        ambient_new=args.ambient_new,
        strength=args.strength,
        boundary_relight=args.boundary_relight,
        shadow_relight=args.shadow_relight,
        specular_strength=args.highlight,
    )

    try:
        result = run_auto_relight(
            image_path=Path(args.image),
            output_root=Path(args.output_root),
            dsine_root=Path(args.dsine_root),
            dsine_python=(
                Path(args.dsine_python)
                if args.dsine_python
                else None
            ),
            device=args.device,
            job_id=args.job_id,
            save_debug=args.save_debug,
            settings=settings,
            overwrite=args.overwrite,
        )

    except PipelineError as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": error.stage,
                    "message": error.message,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
