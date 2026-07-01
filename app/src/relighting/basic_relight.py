import argparse
from pathlib import Path

import numpy as np
from PIL import Image



def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Convert any vector into a unit vector.

    Need:
    Light direction and normals must have length 1.
    Otherwise dot product lighting becomes mathematically wrong.

    Implementation:
    Divide the vector by its length.
    """
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError("Vector length is zero. Light direction cannot be [0, 0, 0].")

    return vector / length


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """
    Convert normal display RGB into linear RGB.

    Need:
    Most images are stored in sRGB, which is not mathematically linear.
    Lighting calculations should happen in linear color space.

    Implementation:
    Standard sRGB to linear conversion.
    """
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB back to display-ready sRGB.

    Need:
    After lighting math, we need to save a normal image that looks correct
    in image viewers.

    Implementation:
    Standard linear to sRGB conversion.
    """
    image = np.clip(image, 0.0, 1.0)

    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * (image ** (1 / 2.4)) - 0.055,
    )


def load_rgb_image(path: Path) -> np.ndarray:
    """
    Load image as float RGB in range [0, 1].
    """
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def load_normal_map(path: Path, flip_y: bool = False, invert_z: bool = False) -> np.ndarray:
    """
    Load DSINE normal map and decode it.

    Need:
    Normal maps are usually saved as RGB colors.
    But for lighting, we need real 3D normal vectors.

    Normal encoding:
    RGB [0, 255] becomes XYZ [-1, +1]

    Example:
    128 roughly means 0
    255 means +1
    0 means -1
    """
    normal_rgb = load_rgb_image(path)

    normals = normal_rgb * 2.0 - 1.0

    if flip_y:
        normals[..., 1] *= -1.0

    if invert_z:
        normals[..., 2] *= -1.0

    length = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(length, 1e-8)

    return normals


def resize_to_match(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize image to match normal map size if needed.

    Need:
    Image and normal map must have exactly same width and height.
    """
    target_h, target_w = target_shape

    pil_image = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    pil_image = pil_image.resize((target_w, target_h), Image.BICUBIC)

    return np.asarray(pil_image).astype(np.float32) / 255.0


def relight_image(
    image: np.ndarray,
    normals: np.ndarray,
    light_direction: np.ndarray,
    ambient: float,
    diffuse_strength: float,
    specular_strength: float,
    shininess: float,
    exposure: float,
    strength: float,
) -> np.ndarray:
    """
    Apply simple physically-inspired relighting.

    Need:
    This is the core of Sunit stage 3.
    It uses the normal map to decide brightness pixel-by-pixel.

    Implementation:
    1. Convert image to linear RGB.
    2. Calculate dot(normal, light_direction).
    3. Use dot product as diffuse light strength.
    4. Add ambient light so dark areas are not completely black.
    5. Add optional specular highlight.
    6. Convert back to sRGB.
    """
    image_linear = srgb_to_linear(image)

    light_direction = normalize_vector(light_direction)

    diffuse = np.sum(normals * light_direction.reshape(1, 1, 3), axis=2)
    diffuse = np.clip(diffuse, 0.0, 1.0)

    shading = ambient + diffuse_strength * diffuse
    shading = shading[..., None]

    relit = image_linear * shading

    if specular_strength > 0:
        view_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vector = normalize_vector(light_direction + view_direction)

        specular = np.sum(normals * half_vector.reshape(1, 1, 3), axis=2)
        specular = np.clip(specular, 0.0, 1.0) ** shininess
        specular = specular[..., None] * specular_strength

        relit = relit + specular

    relit = relit * exposure
    relit_srgb = linear_to_srgb(relit)
    strength = np.clip(strength, 0.0, 1.0)
    final = image * (1.0 - strength) + relit_srgb * strength

    return np.clip(final, 0.0, 1.0)


def save_image(image: np.ndarray, path: Path) -> None:
    """
    Save float RGB image in range [0, 1] as PNG/JPG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic normal-map-based relighting for Sunit.")

    parser.add_argument("--image", required=True, type=Path, help="Input RGB image path.")
    parser.add_argument("--normal", required=True, type=Path, help="Input normal map path.")
    parser.add_argument("--output", required=True, type=Path, help="Output relit image path.")

    parser.add_argument("--light-x", type=float, default=-0.5)
    parser.add_argument("--light-y", type=float, default=-0.3)
    parser.add_argument("--light-z", type=float, default=0.8)

    parser.add_argument("--ambient", type=float, default=0.45)
    parser.add_argument("--diffuse-strength", type=float, default=0.85)
    parser.add_argument("--specular-strength", type=float, default=0.08)
    parser.add_argument("--shininess", type=float, default=32.0)
    parser.add_argument("--exposure", type=float, default=1.0)

    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--invert-z", action="store_true")
    parser.add_argument("--strength", type=float, default=0.35)

    args = parser.parse_args()

    image = load_rgb_image(args.image)
    normals = load_normal_map(args.normal, flip_y=args.flip_y, invert_z=args.invert_z)

    if image.shape[:2] != normals.shape[:2]:
        print("Image and normal map sizes do not match. Resizing image to match normal map.")
        image = resize_to_match(image, normals.shape[:2])

    light_direction = np.array(
        [args.light_x, args.light_y, args.light_z],
        dtype=np.float32,
    )

    relit = relight_image(
        image=image,
        normals=normals,
        light_direction=light_direction,
        ambient=args.ambient,
        diffuse_strength=args.diffuse_strength,
        specular_strength=args.specular_strength,
        shininess=args.shininess,
        exposure=args.exposure,
        strength=args.strength
    )

    save_image(relit, args.output)

    print(f"Relit image saved to: {args.output}")


if __name__ == "__main__":
    main()