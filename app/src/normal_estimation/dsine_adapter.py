from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


class DSINEError(RuntimeError):
    """Raised when DSINE inference fails."""


class DSINEAdapter:
    """
    Adapter around the official DSINE sample inference script.

    This keeps DSINE-specific paths outside the rest of Sunit.
    Later, this adapter can be replaced by direct model loading
    without changing the relighting pipeline.
    """

    def __init__(
        self,
        dsine_root: Path,
        python_executable: Path | None = None,
    ) -> None:
        self.dsine_root = dsine_root.resolve()

        self.project_dir = (
            self.dsine_root
            / "projects"
            / "dsine"
        )

        self.input_dir = (
            self.project_dir
            / "samples"
            / "img"
        )

        self.output_dir = (
            self.project_dir
            / "samples"
            / "output"
        )

        # Relative path because test_minimal.py is run from projects/dsine.
        self.config_path = (
            Path("experiments")
            / "exp001_cvpr2024"
            / "dsine.txt"
        )

        self.absolute_config_path = (
            self.project_dir
            / self.config_path
        )

        self.python_executable = (
            python_executable.expanduser().absolute()
            if python_executable is not None
            else Path(sys.executable).absolute()
        )

    def validate_installation(self) -> None:
        required_paths = [
            self.dsine_root,
            self.project_dir,
            self.input_dir,
            self.output_dir,
            self.absolute_config_path,
            (
                self.project_dir
                / "checkpoints"
                / "exp001_cvpr2024"
                / "dsine.pt"
            ),
            self.project_dir / "test_minimal.py",
            self.dsine_root / "utils" / "utils.py",
            self.dsine_root / "models" / "dsine" / "v02.py",
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            formatted = "\n".join(
                f"  - {path}"
                for path in missing_paths
            )

            raise DSINEError(
                "DSINE installation is incomplete.\n"
                f"DSINE root: {self.dsine_root}\n"
                f"Python executable: {self.python_executable}\n"
                f"Working directory: {self.project_dir}\n"
                f"Missing paths:\n{formatted}"
            )

        if not self.python_executable.exists():
            raise DSINEError(
                "Python executable for DSINE does not exist.\n"
                f"Python executable: {self.python_executable}\n"
                "In Docker this should normally be the container Python, "
                "for example /usr/local/bin/python3.10."
            )

    def build_environment(self) -> dict[str, str]:
        """
        Build a clean environment for the DSINE subprocess.

        Important:
        We intentionally do NOT keep the existing PYTHONPATH.
        Your terminal currently contains another project path:
            /home/satyam_singh/adaptive-aging-aware-DNN

        That project has its own 'models' package, which was hijacking:
            from models.dsine.v02 import DSINE_v02

        So here we give DSINE only the paths it needs.
        """

        env = os.environ.copy()

        # DSINE must not inherit unrelated project paths.
        # Otherwise packages such as "models" and "utils" may be
        # imported from another repository.
        dsine_paths = [
            str(self.dsine_root),
            str(self.project_dir),
        ]

        env["PYTHONPATH"] = os.pathsep.join(dsine_paths)
        env["PYTHONNOUSERSITE"] = "1"

        return env

    def estimate(
        self,
        input_image: Path,
        destination: Path,
    ) -> Path:
        self.validate_installation()

        input_image = input_image.resolve()
        destination = destination.resolve()

        if not input_image.exists():
            raise FileNotFoundError(
                f"Input image was not found: {input_image}"
            )

        self.input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        unique_name = (
            f"sunit_{int(time.time() * 1000)}"
            f"{input_image.suffix.lower()}"
        )

        staged_input = self.input_dir / unique_name

        before_outputs = {
            path.resolve()
            for path in self.output_dir.rglob("*")
            if path.is_file()
        }

        shutil.copy2(
            input_image,
            staged_input,
        )

        command = [
            str(self.python_executable),
            "test_minimal.py",
            str(self.config_path),
        ]

        dsine_env = self.build_environment()

        print("Running DSINE inference...")
        print(f"Resolved DSINE root: {self.dsine_root}")
        print(f"Python executable: {self.python_executable}")
        print(f"Working directory: {self.project_dir}")
        print("Command:")
        print(" ".join(command))
        print("PYTHONPATH:")
        print(dsine_env.get("PYTHONPATH", ""))

        completed: subprocess.CompletedProcess[str] | None = None

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.project_dir),
                env=dsine_env,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            staged_input.unlink(missing_ok=True)

        if completed is None:
            raise DSINEError(
                "DSINE subprocess did not start correctly."
            )

        if completed.returncode != 0:
            raise DSINEError(
                "DSINE inference failed.\n\n"
                f"DSINE root: {self.dsine_root}\n"
                f"Python executable: {self.python_executable}\n"
                f"Working directory: {self.project_dir}\n"
                f"PYTHONPATH: {dsine_env.get('PYTHONPATH', '')}\n\n"
                f"Standard output:\n{completed.stdout}\n\n"
                f"Standard error:\n{completed.stderr}"
            )

        after_outputs = {
            path.resolve()
            for path in self.output_dir.rglob("*")
            if path.is_file()
        }

        new_outputs = list(
            after_outputs - before_outputs
        )

        image_outputs = [
            path
            for path in new_outputs
            if path.suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        ]

        if not image_outputs:
            raise DSINEError(
                "DSINE completed but no new normal-map "
                "image was found in the output directory.\n"
                f"Checked directory: {self.output_dir}"
            )

        matching_outputs = [
            path
            for path in image_outputs
            if staged_input.stem in path.stem
        ]

        candidates = (
            matching_outputs
            if matching_outputs
            else image_outputs
        )

        generated_normal = max(
            candidates,
            key=lambda path: path.stat().st_mtime,
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            generated_normal,
            destination,
        )

        print(f"Normal map saved to: {destination}")

        return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a normal map using the "
            "official DSINE implementation."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input photograph.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Destination normal-map path.",
    )

    parser.add_argument(
        "--dsine-root",
        default="../external/DSINE",
        help="Path to the cloned DSINE repository.",
    )

    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Optional path to DSINE's Python executable. "
            "Defaults to the current Python executable."
        ),
    )

    args = parser.parse_args()

    adapter = DSINEAdapter(
        dsine_root=Path(args.dsine_root),
        python_executable=(
            Path(args.python)
            if args.python is not None
            else None
        ),
    )

    adapter.estimate(
        input_image=Path(args.image),
        destination=Path(args.output),
    )


if __name__ == "__main__":
    main()
