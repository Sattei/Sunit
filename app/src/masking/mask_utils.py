from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def read_rgb(path: str | Path) -> np.ndarray:
    """
    Read an image as RGB float32 in [0, 1].
    """
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def save_gray(path: str | Path, mask: np.ndarray) -> None:
    """
    Save a single-channel mask in [0, 1] as an 8-bit grayscale PNG.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.clip(mask, 0.0, 1.0)
    image = (mask * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(image, mode="L").save(path)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """
    Convert sRGB image values to linear RGB.

    Lighting and luminance calculations should happen in linear space.
    """
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    """
    Compute perceptual luminance from linear RGB.
    """
    return (
        0.2126 * linear_rgb[..., 0]
        + 0.7152 * linear_rgb[..., 1]
        + 0.0722 * linear_rgb[..., 2]
    )


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """
    Smooth threshold function.

    Returns 0 below edge0, 1 above edge1, and a smooth transition between.
    """
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gaussian_blur(mask: np.ndarray, radius: float) -> np.ndarray:
    """
    Feather a mask using PIL Gaussian blur.
    """
    mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    image = Image.fromarray(mask_u8, mode="L")
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image).astype(np.float32) / 255.0


def ellipse_mask(
    height: int,
    width: int,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    softness: float = 0.18,
) -> np.ndarray:
    """
    Create a soft ellipse mask.

    cx, cy, rx, ry are normalized values.
    Example:
        cx=0.5 means center horizontally.
        rx=0.35 means 35% of image width.
    """
    y, x = np.mgrid[0:height, 0:width]

    nx = (x - cx * width) / max(rx * width, 1e-8)
    ny = (y - cy * height) / max(ry * height, 1e-8)

    distance = np.sqrt(nx * nx + ny * ny)

    return 1.0 - smoothstep(1.0 - softness, 1.0 + softness, distance)