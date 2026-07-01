from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from normal_estimation.dsine_adapter import DSINEAdapter


def run_relighting(
    image_path: Path,
    normal_path: Path,
    output_path: Path,
    light_x: float,
    light_y: float,
    light_z: float,
    intensity: float,
    ambient: float,
    temperature: float,
) -> None:
    relight_script = (
        Path(__file__).resolve().parent
        / "relight.py"
    )

    command = [
        sys.executable,
        str(relight_script),
        "--image",
        str(image_path),
        "--normal",
        str(normal_path),
        "--output",
        str(output_path),
        "--light-x",
        str(light_x),
        "--light-y",
        str(light_y),
        "--light-z",
        str(light_z),
        "--intensity",
        str(intensity),
        "--ambient",
        str(ambient),
        "--temperature",
        str(temperature),
        "--flip-y",
    ]

    print()
    print("Running Sunit relighting...")
    print(" ".join(command))

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "The relighting stage failed."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sunit Stage 2: estimate normals and "
            "relight a single image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--dsine-root",
        default="../external/DSINE",
    )

    parser.add_argument(
        "--normal-output",
        default=None,
    )

    parser.add_argument(
        "--reuse-normal",
        action="store_true",
        help=(
            "Reuse an existing normal map instead "
            "of running DSINE again."
        ),
    )

    parser.add_argument(
        "--light-x",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--light-y",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--light-z",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--intensity",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--ambient",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=4500.0,
    )

    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if args.normal_output:
        normal_path = Path(
            args.normal_output
        ).resolve()
    else:
        normal_path = (
            Path("data/normals")
            / f"{image_path.stem}_dsine.png"
        ).resolve()

    if args.reuse_normal:
        if not normal_path.exists():
            raise FileNotFoundError(
                "--reuse-normal was supplied, but "
                f"the normal map does not exist: {normal_path}"
            )

        print(
            f"Reusing existing normal map: {normal_path}"
        )
    else:
        adapter = DSINEAdapter(
            dsine_root=Path(
                args.dsine_root
            ),
        )

        adapter.estimate(
            input_image=image_path,
            destination=normal_path,
        )

    run_relighting(
        image_path=image_path,
        normal_path=normal_path,
        output_path=output_path,
        light_x=args.light_x,
        light_y=args.light_y,
        light_z=args.light_z,
        intensity=args.intensity,
        ambient=args.ambient,
        temperature=args.temperature,
    )

    print()
    print("Sunit Stage 2 completed.")
    print(f"Original image: {image_path}")
    print(f"Normal map: {normal_path}")
    print(f"Relit output: {output_path}")


if __name__ == "__main__":
    main()