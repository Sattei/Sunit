import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# -----------------------------
# Basic image utilities
# -----------------------------

def load_rgb(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def save_rgb(path: str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def save_gray(path: str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    image_uint8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(path)


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
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError("Light direction cannot be zero.")

    return vector / length


# -----------------------------
# Normal map handling
# -----------------------------

def load_normal_map(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    normal_rgb = load_rgb(path)

    h, w = target_hw
    if normal_rgb.shape[:2] != (h, w):
        normal_rgb = cv2.resize(
            normal_rgb,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    # DSINE-style normal visualization usually stores normals in RGB [0, 1].
    # Convert to [-1, 1].
    normal = normal_rgb * 2.0 - 1.0

    # Normalize every vector again for safety.
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(norm, 1e-6)

    return normal.astype(np.float32)


# -----------------------------
# Mask helpers
# -----------------------------

def blur_mask(mask: np.ndarray, kernel: int = 31) -> np.ndarray:
    kernel = max(3, kernel)
    if kernel % 2 == 0:
        kernel += 1

    blurred = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return np.clip(blurred, 0.0, 1.0)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def luminance(image_linear: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image_linear[:, :, 0]
        + 0.7152 * image_linear[:, :, 1]
        + 0.0722 * image_linear[:, :, 2]
    )


def center_prior(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0

    # Portraits usually have subject near center.
    distance = np.sqrt((x / 0.85) ** 2 + (y / 1.05) ** 2)
    prior = 1.0 - smoothstep(0.25, 1.15, distance)
    return np.clip(prior, 0.0, 1.0)


def build_skin_like_mask(image_srgb: np.ndarray) -> np.ndarray:
    """
    Lightweight face/skin-ish protection mask.
    This is not identity detection. It only detects warm skin-colored zones.
    Used to avoid harsh face darkening/highlighting.
    """
    image_uint8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2YCrCb)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    hsv_skin = (
        (h >= 0) & (h <= 30) &
        (s >= 25) & (s <= 180) &
        (v >= 40)
    )

    ycrcb_skin = (
        (cr >= 133) & (cr <= 180) &
        (cb >= 75) & (cb <= 140) &
        (y >= 35)
    )

    mask = (hsv_skin & ycrcb_skin).astype(np.float32)
    mask = blur_mask(mask, 21)
    mask = smoothstep(0.15, 0.65, mask)

    return mask


def load_depth_or_build_pseudo(
    depth_path: str | None,
    image_linear: np.ndarray,
    person_hint: np.ndarray,
) -> np.ndarray:
    """
    Depth convention:
    1.0 = closer to camera / foreground
    0.0 = farther background

    If an external depth map is not given, we create a pseudo-depth using:
    - person/center prior
    - brightness structure
    This is not real depth, but helps reduce background over-relighting.
    """
    h, w = image_linear.shape[:2]

    if depth_path:
        depth_img = Image.open(depth_path).convert("L")
        depth = np.asarray(depth_img).astype(np.float32) / 255.0

        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        depth = cv2.GaussianBlur(depth, (31, 31), 0)
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
        return np.clip(depth, 0.0, 1.0)

    luma = luminance(image_linear)
    luma_smooth = cv2.GaussianBlur(luma, (41, 41), 0)

    cprior = center_prior(h, w)

    pseudo = (
        0.60 * person_hint +
        0.30 * cprior +
        0.10 * smoothstep(0.08, 0.75, luma_smooth)
    )

    pseudo = cv2.GaussianBlur(pseudo, (41, 41), 0)
    pseudo = (pseudo - pseudo.min()) / max(pseudo.max() - pseudo.min(), 1e-6)

    return np.clip(pseudo, 0.0, 1.0)


def build_region_masks(image_srgb: np.ndarray, image_linear: np.ndarray, depth: np.ndarray):
    h, w = image_srgb.shape[:2]

    cprior = center_prior(h, w)
    skin_mask = build_skin_like_mask(image_srgb)

    # Foreground estimate from depth + center prior + skin.
    foreground = (
        0.65 * smoothstep(0.35, 0.85, depth) +
        0.25 * cprior +
        0.10 * skin_mask
    )
    foreground = blur_mask(foreground, 35)
    foreground = smoothstep(0.25, 0.75, foreground)

    background = 1.0 - foreground
    background = blur_mask(background, 35)

    # Face/skin protection should mostly affect foreground.
    face_protect = np.clip(skin_mask * foreground, 0.0, 1.0)
    face_protect = blur_mask(face_protect, 25)

    # Dark original shadow areas should not become crushed black.
    luma = luminance(image_linear)
    dark_protect = 1.0 - smoothstep(0.03, 0.22, luma)
    dark_protect = blur_mask(dark_protect, 25)

    return {
        "foreground": foreground,
        "background": background,
        "face_protect": face_protect,
        "dark_protect": dark_protect,
    }


# -----------------------------
# Relighting core
# -----------------------------

def compute_lambert(normal: np.ndarray, light: np.ndarray) -> np.ndarray:
    light = normalize_vector(light)
    shading = np.sum(normal * light.reshape(1, 1, 3), axis=2)
    return np.clip(shading, 0.0, 1.0)


def compute_half_vector_highlight(
    normal: np.ndarray,
    light: np.ndarray,
    shininess: float,
) -> np.ndarray:
    """
    Simple view-dependent highlight.
    Camera/view direction is approximated as [0, 0, 1].
    """
    light = normalize_vector(light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    half_vec = normalize_vector(light + view)
    ndoth = np.sum(normal * half_vec.reshape(1, 1, 3), axis=2)
    ndoth = np.clip(ndoth, 0.0, 1.0)

    highlight = np.power(ndoth, shininess)
    return np.clip(highlight, 0.0, 1.0)


def stage5_relight(
    image_srgb: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    old_light: np.ndarray,
    new_light: np.ndarray,
    person_strength: float,
    background_strength: float,
    ambient: float,
    highlight_strength: float,
    highlight_shininess: float,
    exposure: float,
    contrast: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image_linear = srgb_to_linear(image_srgb)

    masks = build_region_masks(image_srgb, image_linear, depth)

    foreground = masks["foreground"]
    background = masks["background"]
    face_protect = masks["face_protect"]
    dark_protect = masks["dark_protect"]

    old_shading = compute_lambert(normal, old_light)
    new_shading = compute_lambert(normal, new_light)

    # Ambient fill prevents crushed black shadows.
    old_total = ambient + (1.0 - ambient) * old_shading
    new_total = ambient + (1.0 - ambient) * new_shading

    ratio = new_total / np.maximum(old_total, 0.08)

    # Clamp ratio to prevent extreme artifacts.
    ratio = np.clip(ratio, 0.55, 1.75)

    # Region-aware strength:
    # - foreground gets visible relighting
    # - background gets much less
    # - face gets softened
    strength_map = (
        person_strength * foreground +
        background_strength * background
    )

    # Protect face from harsh change.
    strength_map *= (1.0 - 0.45 * face_protect)

    # Protect already-dark areas from becoming dirty/black.
    strength_map *= (1.0 - 0.55 * dark_protect * background)

    strength_map = blur_mask(strength_map, 17)
    strength_map_3 = strength_map[:, :, None]

    relit_linear = image_linear * (1.0 + (ratio[:, :, None] - 1.0) * strength_map_3)

    # Controlled highlight. Mostly foreground, reduced on background.
    highlight = compute_half_vector_highlight(normal, new_light, highlight_shininess)

    highlight_mask = foreground * (1.0 - 0.65 * face_protect)
    highlight_mask = blur_mask(highlight_mask, 21)

    highlight_amount = highlight_strength * highlight * highlight_mask
    relit_linear = relit_linear + highlight_amount[:, :, None]

    # Gentle exposure/contrast finishing in linear space.
    relit_linear = relit_linear * exposure
    relit_linear = np.clip(relit_linear, 0.0, 1.0)

    if abs(contrast - 1.0) > 1e-6:
        relit_linear = 0.5 + (relit_linear - 0.5) * contrast

    relit_srgb = linear_to_srgb(relit_linear)
    relit_srgb = np.clip(relit_srgb, 0.0, 1.0)

    debug = {
        "depth": depth,
        "foreground": foreground,
        "background": background,
        "face_protect": face_protect,
        "dark_protect": dark_protect,
        "old_shading": old_shading,
        "new_shading": new_shading,
        "ratio": (ratio - 0.55) / (1.75 - 0.55),
        "strength": strength_map,
        "highlight": highlight,
    }

    return relit_srgb, debug


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 5 Sunit relighting: depth-aware + face/background protected relighting."
    )

    parser.add_argument("--image", required=True, help="Input RGB image path.")
    parser.add_argument("--normal", required=True, help="DSINE normal map path.")
    parser.add_argument("--depth", default=None, help="Optional depth map path. White/bright = closer.")
    parser.add_argument("--output", required=True, help="Output relit image path.")

    parser.add_argument(
        "--old-light",
        nargs=3,
        type=float,
        default=[-0.25, -0.15, 1.0],
        metavar=("X", "Y", "Z"),
        help="Estimated original light direction.",
    )

    parser.add_argument(
        "--new-light",
        nargs=3,
        type=float,
        default=[0.55, -0.20, 0.80],
        metavar=("X", "Y", "Z"),
        help="Target new light direction.",
    )

    parser.add_argument(
        "--person-strength",
        type=float,
        default=0.75,
        help="Relighting strength on foreground/person.",
    )

    parser.add_argument(
        "--background-strength",
        type=float,
        default=0.18,
        help="Relighting strength on background.",
    )

    parser.add_argument(
        "--ambient",
        type=float,
        default=0.36,
        help="Ambient fill amount. Higher prevents crushed shadows.",
    )

    parser.add_argument(
        "--highlight",
        type=float,
        default=0.12,
        help="Specular/highlight strength.",
    )

    parser.add_argument(
        "--highlight-shininess",
        type=float,
        default=48.0,
        help="Higher = tighter highlight.",
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="Final exposure multiplier.",
    )

    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Final contrast multiplier.",
    )

    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug masks and maps next to output.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_srgb = load_rgb(args.image)
    h, w = image_srgb.shape[:2]

    normal = load_normal_map(args.normal, (h, w))

    # Initial rough person hint before depth.
    image_linear = srgb_to_linear(image_srgb)
    skin_mask = build_skin_like_mask(image_srgb)
    person_hint = np.clip(0.65 * center_prior(h, w) + 0.35 * skin_mask, 0.0, 1.0)

    depth = load_depth_or_build_pseudo(
        args.depth,
        image_linear=image_linear,
        person_hint=person_hint,
    )

    output, debug = stage5_relight(
        image_srgb=image_srgb,
        normal=normal,
        depth=depth,
        old_light=np.array(args.old_light, dtype=np.float32),
        new_light=np.array(args.new_light, dtype=np.float32),
        person_strength=args.person_strength,
        background_strength=args.background_strength,
        ambient=args.ambient,
        highlight_strength=args.highlight,
        highlight_shininess=args.highlight_shininess,
        exposure=args.exposure,
        contrast=args.contrast,
    )

    save_rgb(args.output, output)
    print(f"Stage 5 relit image saved to: {args.output}")

    if args.save_debug:
        output_path = Path(args.output)
        debug_dir = output_path.parent / f"{output_path.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for name, value in debug.items():
            save_gray(debug_dir / f"{name}.png", value)

        print(f"Debug maps saved to: {debug_dir}")


if __name__ == "__main__":
    main()