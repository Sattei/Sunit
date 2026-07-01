import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_flat_normal(width: int, height: int) -> np.ndarray:
    """
    Create a normal map where every surface faces directly toward the camera.

    Normal direction:
        X = 0
        Y = 0
        Z = 1
    """
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    return normals


def create_sphere_normal(width: int, height: int) -> np.ndarray:
    """
    Create a sphere-shaped normal map.

    This does not represent the real geometry of the input photograph.
    It is only used to verify that changing the light direction works.
    """
    x_coordinates = np.linspace(-1.0, 1.0, width)
    y_coordinates = np.linspace(1.0, -1.0, height)

    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)

    radius_squared = x_grid**2 + y_grid**2
    inside_sphere = radius_squared <= 1.0

    normal_x = np.zeros_like(x_grid, dtype=np.float32)
    normal_y = np.zeros_like(y_grid, dtype=np.float32)
    normal_z = np.ones_like(x_grid, dtype=np.float32)

    normal_x[inside_sphere] = x_grid[inside_sphere]
    normal_y[inside_sphere] = y_grid[inside_sphere]
    normal_z[inside_sphere] = np.sqrt(
        1.0 - radius_squared[inside_sphere]
    )

    normals = np.stack(
        [normal_x, normal_y, normal_z],
        axis=-1,
    )

    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)

    return normals / lengths


def encode_normal_map(normals: np.ndarray) -> np.ndarray:
    """
    Convert normal values from [-1, 1] into image values [0, 255].
    """
    encoded = (normals + 1.0) / 2.0
    encoded = np.clip(encoded, 0.0, 1.0)

    return (encoded * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a test normal map for Sunit."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Input image whose dimensions should be used.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Location where the normal map will be saved.",
    )

    parser.add_argument(
        "--mode",
        choices=["flat", "sphere"],
        default="sphere",
        help="Type of test normal map to generate.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image was not found: {image_path}"
        )

    with Image.open(image_path) as image:
        width, height = image.size

    if args.mode == "flat":
        normals = create_flat_normal(width, height)
    else:
        normals = create_sphere_normal(width, height)

    encoded_normal_map = encode_normal_map(normals)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(
        encoded_normal_map,
        mode="RGB",
    ).save(output_path)

    print(f"Normal map created: {output_path}")
    print(f"Mode: {args.mode}")
    print(f"Resolution: {width} × {height}")


if __name__ == "__main__":
    main()