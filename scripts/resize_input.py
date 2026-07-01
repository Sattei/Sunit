import argparse
from pathlib import Path

from PIL import Image, ImageOps


def resize_image(
    input_path: Path,
    output_path: Path,
    maximum_side: int,
) -> None:
    with Image.open(input_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        width, height = image.size

        scale = min(
            1.0,
            maximum_side / max(width, height),
        )

        resized_width = round(width * scale)
        resized_height = round(height * scale)

        resized = image.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        resized.save(
            output_path,
            quality=95,
        )

    print(
        f"Resized {width} × {height} "
        f"to {resized_width} × {resized_height}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
    )

    args = parser.parse_args()

    resize_image(
        input_path=Path(args.input),
        output_path=Path(args.output),
        maximum_side=args.max_side,
    )


if __name__ == "__main__":
    main()