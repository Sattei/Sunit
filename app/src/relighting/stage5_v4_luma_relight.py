from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# IO + color
# ============================================================

def load_rgb(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def save_rgb(path: str, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


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


def luminance(image_linear: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image_linear[:, :, 0]
        + 0.7152 * image_linear[:, :, 1]
        + 0.0722 * image_linear[:, :, 2]
    )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError("Light direction cannot be zero.")

    return vector / length


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def odd_kernel(value: int) -> int:
    value = max(3, int(value))
    if value % 2 == 0:
        value += 1
    return value


def blur(mask: np.ndarray, kernel: int) -> np.ndarray:
    kernel = odd_kernel(kernel)
    out = cv2.GaussianBlur(mask.astype(np.float32), (kernel, kernel), 0)
    return np.clip(out, 0.0, 1.0)


# ============================================================
# Normals
# ============================================================

def load_normal_map(
    path: str,
    target_hw: tuple[int, int],
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    normal_rgb = load_rgb(path)

    h, w = target_hw

    if normal_rgb.shape[:2] != (h, w):
        normal_rgb = cv2.resize(normal_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    normal = normal_rgb * 2.0 - 1.0

    if flip_x:
        normal[:, :, 0] *= -1.0
    if flip_y:
        normal[:, :, 1] *= -1.0
    if flip_z:
        normal[:, :, 2] *= -1.0

    length = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / np.maximum(length, 1e-6)

    return normal.astype(np.float32)


def lambert(normal: np.ndarray, light: np.ndarray) -> np.ndarray:
    light = normalize_vector(light)
    shading = np.sum(normal * light.reshape(1, 1, 3), axis=2)
    return np.clip(shading, 0.0, 1.0).astype(np.float32)


def smooth_shading(shading: np.ndarray) -> np.ndarray:
    shading = cv2.GaussianBlur(shading.astype(np.float32), (61, 61), 0)

    shading = cv2.bilateralFilter(
        shading.astype(np.float32),
        d=13,
        sigmaColor=0.10,
        sigmaSpace=45,
    )

    return np.clip(shading, 0.0, 1.0)


# ============================================================
# Masks
# ============================================================

def center_prior(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0

    distance = np.sqrt((x / 0.95) ** 2 + ((y + 0.12) / 1.15) ** 2)
    prior = 1.0 - smoothstep(0.20, 1.15, distance)

    return np.clip(prior, 0.0, 1.0)


def build_skin_mask(image_srgb: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)

    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    hsv_skin = (
        (h >= 0) & (h <= 32) &
        (s >= 18) & (s <= 200) &
        (v >= 40)
    )

    ycrcb_skin = (
        (cr >= 130) & (cr <= 188) &
        (cb >= 68) & (cb <= 150) &
        (y >= 30)
    )

    mask = (hsv_skin & ycrcb_skin).astype(np.float32)
    mask = blur(mask, 25)
    mask = smoothstep(0.08, 0.55, mask)

    return np.clip(mask, 0.0, 1.0)


def load_optional_mask(path: str | None, target_hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None

    h, w = target_hw

    mask = Image.open(path).convert("L")
    mask = np.asarray(mask).astype(np.float32) / 255.0

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = blur(mask, 25)
    mask = smoothstep(0.10, 0.85, mask)

    return np.clip(mask, 0.0, 1.0)


def build_auto_foreground(image_srgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Soft automatic foreground estimate.
    Not perfect, but safer than relighting everything.
    """
    h, w = image_srgb.shape[:2]

    cprior = center_prior(h, w)

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = x / max(w - 1, 1)
    y_norm = y / max(h - 1, 1)

    body_prior = (
        smoothstep(0.10, 0.35, x_norm)
        * (1.0 - smoothstep(0.92, 1.00, x_norm))
        * smoothstep(0.22, 0.50, y_norm)
    )

    foreground = (
        0.48 * cprior +
        0.34 * body_prior +
        0.18 * skin
    )

    foreground = blur(foreground, 51)
    foreground = smoothstep(0.28, 0.72, foreground)

    return np.clip(foreground, 0.0, 1.0)


def build_shadow_mask(image_srgb: np.ndarray, image_linear: np.ndarray) -> np.ndarray:
    """
    White = existing dark/shadow region.
    This is used to avoid ratio distortion, not to completely freeze everything.
    """
    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)

    value = hsv[:, :, 2].astype(np.float32) / 255.0
    lab_l = lab[:, :, 0].astype(np.float32) / 255.0
    y = luminance(image_linear)

    tone = 0.45 * value + 0.35 * lab_l + 0.20 * np.sqrt(np.clip(y, 0.0, 1.0))

    small = cv2.GaussianBlur(tone, (7, 7), 0)
    mid = cv2.GaussianBlur(tone, (71, 71), 0)
    large = cv2.GaussianBlur(tone, (151, 151), 0)

    abs_dark = 1.0 - smoothstep(0.16, 0.48, small)

    rel_mid = np.maximum(mid - small, 0.0) / np.maximum(mid, 0.08)
    rel_large = np.maximum(large - small, 0.0) / np.maximum(large, 0.08)

    local_shadow = np.maximum(
        smoothstep(0.06, 0.30, rel_mid),
        smoothstep(0.04, 0.24, rel_large),
    )

    shadow = np.maximum(0.55 * abs_dark, local_shadow)

    bright_reject = smoothstep(0.60, 0.82, small)
    shadow *= 1.0 - 0.70 * bright_reject

    shadow = blur(shadow, 31)
    shadow = smoothstep(0.12, 0.82, shadow)

    return np.clip(shadow, 0.0, 1.0)


def build_normal_edge_mask(normal: np.ndarray) -> np.ndarray:
    edge = np.zeros(normal.shape[:2], dtype=np.float32)

    for c in range(3):
        gx = cv2.Sobel(normal[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normal[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        edge += gx * gx + gy * gy

    edge = np.sqrt(edge)
    scale = np.percentile(edge, 95)
    edge = edge / max(scale, 1e-6)

    edge = smoothstep(0.22, 0.90, edge)
    edge = blur(edge, 17)

    return np.clip(edge, 0.0, 1.0)


# ============================================================
# Luminance-only relighting
# ============================================================

def normalize_delta(delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
    safe = mask > 0.25

    if np.count_nonzero(safe) > 50:
        scale = np.percentile(np.abs(delta[safe]), 92)
    else:
        scale = np.percentile(np.abs(delta), 92)

    delta = delta / max(scale, 1e-6)
    delta = np.clip(delta, -1.0, 1.0)
    delta = cv2.GaussianBlur(delta.astype(np.float32), (41, 41), 0)

    return np.clip(delta, -1.0, 1.0)


def apply_luma_scale(
    image_linear: np.ndarray,
    target_luma: np.ndarray,
) -> np.ndarray:
    old_luma = luminance(image_linear)

    scale = target_luma / np.maximum(old_luma, 1e-5)
    scale = np.clip(scale, 0.72, 1.38)

    output = image_linear * scale[:, :, None]

    return np.clip(output, 0.0, 1.0)


def stage5_v4_luma_relight(
    image_srgb: np.ndarray,
    normal: np.ndarray,
    foreground_mask: np.ndarray | None,
    old_light: np.ndarray,
    new_light: np.ndarray,
    strength: float,
    shadow_fill: float,
    max_brighten: float,
    max_darken: float,
    background_strength: float,
    exposure: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image_linear = srgb_to_linear(image_srgb)
    h, w = image_srgb.shape[:2]

    skin = build_skin_mask(image_srgb)

    if foreground_mask is None:
        foreground = build_auto_foreground(image_srgb, skin)
    else:
        foreground = foreground_mask

    background = 1.0 - foreground
    background = blur(background, 35)

    shadow = build_shadow_mask(image_srgb, image_linear)
    edge = build_normal_edge_mask(normal)

    old_raw = lambert(normal, old_light)
    new_raw = lambert(normal, new_light)

    old_smooth = smooth_shading(old_raw)
    new_smooth = smooth_shading(new_raw)

    raw_delta = new_smooth - old_smooth
    delta = normalize_delta(raw_delta, foreground)

    positive_delta = np.maximum(delta, 0.0)
    negative_delta = np.minimum(delta, 0.0)

    permission = foreground.copy()

    # Skin is allowed, but gently.
    permission *= 1.0 - 0.35 * skin

    # Shadows are allowed only for soft fill, not directional distortion.
    permission *= 1.0 - 0.68 * shadow

    # Normal edges are unreliable.
    permission *= 1.0 - 0.65 * edge

    # Background is almost locked.
    permission *= 1.0 - 0.95 * background
    permission += background_strength * background

    permission = blur(permission, 31)
    permission = np.clip(permission, 0.0, 1.0)

    brighten_log = positive_delta * permission * strength * max_brighten
    darken_log = negative_delta * permission * strength * max_darken

    log_gain = brighten_log + darken_log

    old_luma = luminance(image_linear)
    target_luma = old_luma * np.exp(log_gain)

    # Separate shadow fill.
    # This brightens old shadows softly without pretending the normal map can fix them.
    fill_region = foreground * shadow
    fill_region *= 1.0 - 0.60 * background
    fill_region *= 1.0 - 0.35 * edge
    fill_region = blur(fill_region, 35)

    shadow_lift = shadow_fill * fill_region * np.power(1.0 - old_luma, 1.15)
    target_luma = target_luma + shadow_lift * (1.0 - target_luma)

    # Safety: do not let output become much darker than original.
    min_luma = old_luma * (0.92 - 0.12 * permission)
    target_luma = np.maximum(target_luma, min_luma)

    target_luma = np.clip(target_luma, 0.0, 1.0)

    relit_linear = apply_luma_scale(image_linear, target_luma)

    # Preserve highly risky regions, but less aggressively than V3.
    preserve = np.maximum.reduce([
        0.90 * background,
        0.35 * shadow,
        0.45 * edge,
    ])

    # Do not over-preserve skin; otherwise face never changes.
    preserve *= 1.0 - 0.20 * skin

    preserve = blur(preserve, 21)
    preserve = np.clip(preserve, 0.0, 0.92)

    relit_linear = (
        relit_linear * (1.0 - preserve[:, :, None]) +
        image_linear * preserve[:, :, None]
    )

    relit_linear *= exposure
    relit_linear = np.clip(relit_linear, 0.0, 1.0)

    output_srgb = linear_to_srgb(relit_linear)
    output_srgb = np.clip(output_srgb, 0.0, 1.0)

    debug = {
        "foreground": foreground,
        "background": background,
        "skin": skin,
        "shadow": shadow,
        "normal_edge": edge,
        "old_smooth": old_smooth,
        "new_smooth": new_smooth,
        "delta": (delta + 1.0) * 0.5,
        "permission": permission,
        "fill_region": fill_region,
        "preserve": preserve,
        "target_luma": target_luma,
    }

    return output_srgb, debug


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sunit Stage 5 V4: luminance-only shadow-safe relighting."
    )

    parser.add_argument("--image", required=True)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--foreground-mask", default=None)

    parser.add_argument(
        "--old-light",
        nargs=3,
        type=float,
        default=[-0.25, -0.15, 1.0],
    )

    parser.add_argument(
        "--new-light",
        nargs=3,
        type=float,
        default=[0.45, -0.20, 0.88],
    )

    parser.add_argument("--strength", type=float, default=0.65)
    parser.add_argument("--shadow-fill", type=float, default=0.08)

    parser.add_argument("--max-brighten", type=float, default=0.24)
    parser.add_argument("--max-darken", type=float, default=0.04)

    parser.add_argument("--background-strength", type=float, default=0.0)
    parser.add_argument("--exposure", type=float, default=1.0)

    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")

    parser.add_argument("--save-debug", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    image_srgb = load_rgb(args.image)
    h, w = image_srgb.shape[:2]

    normal = load_normal_map(
        path=args.normal,
        target_hw=(h, w),
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )

    foreground_mask = load_optional_mask(
        path=args.foreground_mask,
        target_hw=(h, w),
    )

    output, debug = stage5_v4_luma_relight(
        image_srgb=image_srgb,
        normal=normal,
        foreground_mask=foreground_mask,
        old_light=np.array(args.old_light, dtype=np.float32),
        new_light=np.array(args.new_light, dtype=np.float32),
        strength=args.strength,
        shadow_fill=args.shadow_fill,
        max_brighten=args.max_brighten,
        max_darken=args.max_darken,
        background_strength=args.background_strength,
        exposure=args.exposure,
    )

    save_rgb(args.output, output)
    print(f"Saved Stage 5 V4 output to: {args.output}")

    if args.save_debug:
        output_path = Path(args.output)
        debug_dir = output_path.parent / f"{output_path.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for name, value in debug.items():
            save_gray(debug_dir / f"{name}.png", value)

        print(f"Saved debug maps to: {debug_dir}")


if __name__ == "__main__":
    main()