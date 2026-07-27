"""Archived early ratio-relighting experiment; not used in production."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError("Light direction cannot be [0, 0, 0].")

    return vector / length


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)

    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * (image ** (1 / 2.4)) - 0.055,
    )


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def load_normal_map(path: Path, flip_y: bool = False, invert_z: bool = False) -> np.ndarray:
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
    target_h, target_w = target_shape

    pil_image = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    pil_image = pil_image.resize((target_w, target_h), Image.BICUBIC)

    return np.asarray(pil_image).astype(np.float32) / 255.0


def compute_shading(
    normals: np.ndarray,
    light_direction: np.ndarray,
    ambient: float,
    diffuse_strength: float,
    gamma: float,
) -> np.ndarray:
    light_direction = normalize_vector(light_direction)

    ndotl = np.sum(normals * light_direction.reshape(1, 1, 3), axis=2)
    ndotl = np.clip(ndotl, 0.0, 1.0)

    shading = ambient + diffuse_strength * ndotl

    shading = np.clip(shading, 0.0, 2.0)

    if gamma != 1.0:
        shading = np.power(np.clip(shading, 0.0, 1.0), gamma)

    return shading


def protect_extreme_regions(
    image_linear: np.ndarray,
    ratio: np.ndarray,
    shadow_protect: float,
    highlight_protect: float,
) -> np.ndarray:
    """
    Reduce relighting strength in very dark and very bright regions.

    Need:
    Real photos contain noisy dark shadows and clipped highlights.
    Strong relighting there creates artifacts.

    Implementation:
    Use luminance masks to push ratio back toward 1.0 in risky regions.
    """
    luminance = (
        0.2126 * image_linear[..., 0]
        + 0.7152 * image_linear[..., 1]
        + 0.0722 * image_linear[..., 2]
    )

    dark_mask = 1.0 - np.clip(luminance / 0.25, 0.0, 1.0)
    bright_mask = np.clip((luminance - 0.75) / 0.25, 0.0, 1.0)

    protect_mask = shadow_protect * dark_mask + highlight_protect * bright_mask
    protect_mask = np.clip(protect_mask, 0.0, 1.0)

    ratio = ratio * (1.0 - protect_mask) + 1.0 * protect_mask

    return ratio


def ratio_relight(
    image: np.ndarray,
    normals: np.ndarray,
    old_light: np.ndarray,
    new_light: np.ndarray,
    old_ambient: float,
    old_diffuse: float,
    new_ambient: float,
    new_diffuse: float,
    epsilon: float,
    strength: float,
    ratio_min: float,
    ratio_max: float,
    shadow_protect: float,
    highlight_protect: float,
) -> np.ndarray:
    """
    Ratio-based relighting.

    Need:
    The original image already contains old lighting.
    We estimate old shading and divide it out before applying new shading.

    Implementation:
    final = image * (new_shading / max(old_shading, epsilon))
    Then clamp and blend for stability.
    """
    image_linear = srgb_to_linear(image)

    old_shading = compute_shading(
        normals=normals,
        light_direction=old_light,
        ambient=old_ambient,
        diffuse_strength=old_diffuse,
        gamma=1.0,
    )

    new_shading = compute_shading(
        normals=normals,
        light_direction=new_light,
        ambient=new_ambient,
        diffuse_strength=new_diffuse,
        gamma=1.0,
    )

    ratio = new_shading / np.maximum(old_shading, epsilon)
    ratio = np.clip(ratio, ratio_min, ratio_max)

    ratio = protect_extreme_regions(
        image_linear=image_linear,
        ratio=ratio,
        shadow_protect=shadow_protect,
        highlight_protect=highlight_protect,
    )

    relit_linear = image_linear * ratio[..., None]
    relit_srgb = linear_to_srgb(relit_linear)

    strength = np.clip(strength, 0.0, 1.0)
    final = image * (1.0 - strength) + relit_srgb * strength

    return np.clip(final, 0.0, 1.0)


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ratio-based relighting for Sunit.")

    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--normal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument("--old-light-x", type=float, default=-0.5)
    parser.add_argument("--old-light-y", type=float, default=-0.3)
    parser.add_argument("--old-light-z", type=float, default=0.8)

    parser.add_argument("--new-light-x", type=float, default=0.5)
    parser.add_argument("--new-light-y", type=float, default=-0.3)
    parser.add_argument("--new-light-z", type=float, default=0.8)

    parser.add_argument("--old-ambient", type=float, default=0.30)
    parser.add_argument("--old-diffuse", type=float, default=0.70)

    parser.add_argument("--new-ambient", type=float, default=0.45)
    parser.add_argument("--new-diffuse", type=float, default=0.65)

    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--strength", type=float, default=0.50)

    parser.add_argument("--ratio-min", type=float, default=0.55)
    parser.add_argument("--ratio-max", type=float, default=1.65)

    parser.add_argument("--shadow-protect", type=float, default=0.45)
    parser.add_argument("--highlight-protect", type=float, default=0.35)

    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--invert-z", action="store_true")

    args = parser.parse_args()

    image = load_rgb_image(args.image)
    normals = load_normal_map(
        args.normal,
        flip_y=args.flip_y,
        invert_z=args.invert_z,
    )

    if image.shape[:2] != normals.shape[:2]:
        print("Image and normal map sizes do not match. Resizing image to match normal map.")
        image = resize_to_match(image, normals.shape[:2])

    old_light = np.array(
        [args.old_light_x, args.old_light_y, args.old_light_z],
        dtype=np.float32,
    )

    new_light = np.array(
        [args.new_light_x, args.new_light_y, args.new_light_z],
        dtype=np.float32,
    )

    result = ratio_relight(
        image=image,
        normals=normals,
        old_light=old_light,
        new_light=new_light,
        old_ambient=args.old_ambient,
        old_diffuse=args.old_diffuse,
        new_ambient=args.new_ambient,
        new_diffuse=args.new_diffuse,
        epsilon=args.epsilon,
        strength=args.strength,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        shadow_protect=args.shadow_protect,
        highlight_protect=args.highlight_protect,
    )

    save_image(result, args.output)

    print(f"Ratio relit image saved to: {args.output}")


if __name__ == "__main__":
    main()
