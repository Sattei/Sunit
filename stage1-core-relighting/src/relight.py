import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Convert a vector into a unit vector with length 1.
    """
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError(
            "The light direction cannot be [0, 0, 0]."
        )

    return vector / length


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image into linear RGB.

    Lighting mathematics should be performed in linear RGB rather
    than directly on regular image pixel values.
    """
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a linear RGB image back into display-ready sRGB.
    """
    image = np.clip(image, 0.0, 1.0)

    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def kelvin_to_rgb(temperature: float) -> np.ndarray:
    """
    Approximate a color temperature in Kelvin as RGB.

    Lower values:
        warmer, more orange

    Higher values:
        cooler, more blue
    """
    temperature = float(np.clip(temperature, 1000.0, 40000.0))
    value = temperature / 100.0

    if value <= 66.0:
        red = 255.0
    else:
        red = 329.698727446 * ((value - 60.0) ** -0.1332047592)

    if value <= 66.0:
        green = 99.4708025861 * math.log(value) - 161.1195681661
    else:
        green = 288.1221695283 * ((value - 60.0) ** -0.0755148492)

    if value >= 66.0:
        blue = 255.0
    elif value <= 19.0:
        blue = 0.0
    else:
        blue = 138.5177312231 * math.log(value - 10.0) - 305.044792731

    rgb = np.array(
        [red, green, blue],
        dtype=np.float32,
    )

    rgb = np.clip(rgb, 0.0, 255.0) / 255.0

    return rgb


def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load an image as an RGB float array with values between 0 and 1.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array


def load_normal_map(
    path: Path,
    target_width: int,
    target_height: int,
    flip_y: bool,
) -> np.ndarray:
    """
    Load and decode a normal map.

    The normal map stores values in [0, 255].
    We convert them back into normal values in [-1, 1].
    """
    if not path.exists():
        raise FileNotFoundError(f"Normal map not found: {path}")

    with Image.open(path) as normal_image:
        normal_image = normal_image.convert("RGB")

        if normal_image.size != (target_width, target_height):
            print(
                "Normal map dimensions do not match the image. "
                "Resizing the normal map."
            )

            normal_image = normal_image.resize(
                (target_width, target_height),
                Image.Resampling.BILINEAR,
            )

        encoded = np.asarray(
            normal_image,
            dtype=np.float32,
        ) / 255.0

    normals = encoded * 2.0 - 1.0

    if flip_y:
        normals[..., 1] *= -1.0

    lengths = np.linalg.norm(
        normals,
        axis=-1,
        keepdims=True,
    )

    lengths = np.maximum(lengths, 1e-8)
    normals = normals / lengths

    return normals


def create_relighted_image(
    original_srgb: np.ndarray,
    normals: np.ndarray,
    light_direction: np.ndarray,
    light_intensity: float,
    ambient_strength: float,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply simple Lambertian directional lighting.

    Returns:
        relighted image
        diffuse shading map
    """
    light_direction = normalize_vector(light_direction)

    # Dot product between every normal and the light direction.
    diffuse = np.sum(
        normals * light_direction.reshape(1, 1, 3),
        axis=-1,
    )

    # Surfaces facing away from the light receive no diffuse light.
    diffuse = np.clip(diffuse, 0.0, 1.0)

    light_color_srgb = kelvin_to_rgb(temperature)

    light_color_linear = srgb_to_linear(
        light_color_srgb.reshape(1, 1, 3)
    )

    original_linear = srgb_to_linear(original_srgb)

    neutral_ambient = np.full(
        shape=(1, 1, 3),
        fill_value=ambient_strength,
        dtype=np.float32,
    )

    directional_light = (
        diffuse[..., np.newaxis]
        * light_intensity
        * light_color_linear
    )

    total_lighting = neutral_ambient + directional_light

    relighted_linear = original_linear * total_lighting

    relighted_srgb = linear_to_srgb(relighted_linear)

    return relighted_srgb, diffuse


def save_rgb_image(image: np.ndarray, output_path: Path) -> None:
    """
    Save an RGB float image as an 8-bit PNG or JPEG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

    Image.fromarray(
        encoded,
        mode="RGB",
    ).save(output_path)


def save_grayscale_image(
    image: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a float grayscale image as an 8-bit image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoded = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)

    Image.fromarray(
        encoded,
        mode="L",
    ).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sunit Stage 1 normal-map relighting engine."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the original image.",
    )

    parser.add_argument(
        "--normal",
        required=True,
        help="Path to the normal map.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path where the relit image will be saved.",
    )

    parser.add_argument(
        "--light-x",
        type=float,
        default=0.0,
        help="Horizontal light direction. Positive means right.",
    )

    parser.add_argument(
        "--light-y",
        type=float,
        default=0.0,
        help="Vertical light direction. Positive means up.",
    )

    parser.add_argument(
        "--light-z",
        type=float,
        default=1.0,
        help="Forward light direction. Positive means toward camera.",
    )

    parser.add_argument(
        "--intensity",
        type=float,
        default=1.0,
        help="Strength of the directional light.",
    )

    parser.add_argument(
        "--ambient",
        type=float,
        default=0.3,
        help="Minimum neutral illumination.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=6500.0,
        help="Light color temperature in Kelvin.",
    )

    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Invert the green/Y channel of the normal map.",
    )

    parser.add_argument(
        "--save-shading",
        default=None,
        help="Optional path for saving the grayscale shading map.",
    )

    args = parser.parse_args()

    if args.intensity < 0:
        raise ValueError("Intensity cannot be negative.")

    if args.ambient < 0:
        raise ValueError("Ambient strength cannot be negative.")

    image_path = Path(args.image)
    normal_path = Path(args.normal)
    output_path = Path(args.output)

    original_image = load_rgb_image(image_path)

    height, width, _ = original_image.shape

    normals = load_normal_map(
        path=normal_path,
        target_width=width,
        target_height=height,
        flip_y=args.flip_y,
    )

    light_direction = np.array(
        [
            args.light_x,
            args.light_y,
            args.light_z,
        ],
        dtype=np.float32,
    )

    relighted_image, shading_map = create_relighted_image(
        original_srgb=original_image,
        normals=normals,
        light_direction=light_direction,
        light_intensity=args.intensity,
        ambient_strength=args.ambient,
        temperature=args.temperature,
    )

    save_rgb_image(relighted_image, output_path)

    if args.save_shading:
        save_grayscale_image(
            shading_map,
            Path(args.save_shading),
        )

    normalized_light = normalize_vector(light_direction)

    print()
    print("Sunit Stage 1 relighting completed.")
    print(f"Input image: {image_path}")
    print(f"Normal map: {normal_path}")
    print(f"Output image: {output_path}")
    print(
        "Normalized light direction: "
        f"[{normalized_light[0]:.3f}, "
        f"{normalized_light[1]:.3f}, "
        f"{normalized_light[2]:.3f}]"
    )
    print(f"Intensity: {args.intensity}")
    print(f"Ambient: {args.ambient}")
    print(f"Temperature: {args.temperature} K")


if __name__ == "__main__":
    main()