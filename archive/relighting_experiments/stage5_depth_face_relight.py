"""Archived compatibility copy; production uses src.relighting.engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# IO / Color
# ============================================================

def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8).save(path)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)

    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)

    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def luminance(image_linear: np.ndarray) -> np.ndarray:
    return (
        0.2126 * image_linear[:, :, 0]
        + 0.7152 * image_linear[:, :, 1]
        + 0.0722 * image_linear[:, :, 2]
    ).astype(np.float32)


# ============================================================
# Utilities
# ============================================================

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError("Light direction cannot be [0, 0, 0].")

    return vector / length


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def odd_kernel(value: int) -> int:
    value = max(3, int(round(value)))

    if value % 2 == 0:
        value += 1

    return value


def scaled_kernel(
    base: int,
    height: int,
    width: int,
    ref_side: int = 768,
    min_value: int = 3,
) -> int:
    scale = min(height, width) / float(ref_side)
    return odd_kernel(max(min_value, int(round(base * scale))))


def scaled_size(
    base: int,
    height: int,
    width: int,
    ref_side: int = 768,
    min_value: int = 3,
) -> int:
    scale = min(height, width) / float(ref_side)
    return max(min_value, int(round(base * scale)))


def blur(image: np.ndarray, kernel: int) -> np.ndarray:
    kernel = odd_kernel(kernel)
    return cv2.GaussianBlur(image.astype(np.float32), (kernel, kernel), 0)


def blur01(image: np.ndarray, kernel: int) -> np.ndarray:
    return np.clip(blur(image, kernel), 0.0, 1.0)


def resize_like(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw

    if image.shape[:2] == (h, w):
        return image

    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def binary_keep_largest_components(
    binary: np.ndarray,
    min_area_ratio: float = 0.0004,
    max_components: int = 3,
) -> np.ndarray:
    binary_u8 = (binary > 0).astype(np.uint8)
    h, w = binary_u8.shape
    total_area = h * w

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)

    if num_labels <= 1:
        return binary_u8.astype(np.float32)

    components: list[tuple[int, int]] = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area >= max(20, int(total_area * min_area_ratio)):
            components.append((area, label))

    if not components:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == largest).astype(np.float32)

    components.sort(reverse=True)
    keep_labels = {label for _, label in components[:max_components]}

    kept = np.isin(labels, list(keep_labels)).astype(np.float32)
    return kept


def keep_seeded_components(
    binary: np.ndarray,
    seed: np.ndarray,
    min_area_ratio: float = 0.0004,
) -> np.ndarray:
    binary_u8 = (binary > 0).astype(np.uint8)
    seed_bool = seed > 0.15

    h, w = binary_u8.shape
    total_area = h * w

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)

    if num_labels <= 1:
        return binary_u8.astype(np.float32)

    kept = np.zeros_like(binary_u8)

    largest_label = 0
    largest_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area > largest_area:
            largest_area = area
            largest_label = label

        if area < max(30, int(total_area * min_area_ratio)):
            continue

        component = labels == label
        seed_overlap = int(np.count_nonzero(component & seed_bool))

        if seed_overlap > max(12, int(area * 0.015)):
            kept[component] = 1

    if np.count_nonzero(kept) == 0 and largest_label > 0:
        kept[labels == largest_label] = 1

    return kept.astype(np.float32)


# ============================================================
# Normal map
# ============================================================

def load_normal_map(
    path: str | Path,
    target_hw: tuple[int, int],
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    normal_rgb = load_rgb(path)
    normal_rgb = resize_like(normal_rgb, target_hw)

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
    h, w = shading.shape

    smooth = blur(shading, scaled_kernel(35, h, w))

    d = scaled_size(11, h, w, min_value=5)
    if d % 2 == 0:
        d += 1

    smooth = cv2.bilateralFilter(
        smooth.astype(np.float32),
        d=d,
        sigmaColor=0.08,
        sigmaSpace=scaled_size(35, h, w, min_value=15),
    )

    return np.clip(smooth, 0.0, 1.0)


def build_normal_edge_mask(normal: np.ndarray) -> np.ndarray:
    h, w = normal.shape[:2]
    edge = np.zeros((h, w), dtype=np.float32)

    for c in range(3):
        gx = cv2.Sobel(normal[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normal[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        edge += gx * gx + gy * gy

    edge = np.sqrt(edge)
    scale = np.percentile(edge, 95)
    edge = edge / max(scale, 1e-6)

    edge = smoothstep(0.22, 0.90, edge)
    edge = blur01(edge, scaled_kernel(17, h, w))

    return np.clip(edge, 0.0, 1.0)


def compute_smoothed_diffuse(
    normal: np.ndarray,
    light: np.ndarray,
    normal_edge: np.ndarray,
    skin: np.ndarray,
) -> np.ndarray:
    raw = lambert(normal, light)
    smooth = smooth_shading(raw)

    smooth_weight = np.clip(
        0.20 + 0.45 * skin + 0.60 * normal_edge,
        0.0,
        0.90,
    )

    diffuse = raw * (1.0 - smooth_weight) + smooth * smooth_weight

    skin_soft = smooth_shading(diffuse)
    diffuse = diffuse * (1.0 - 0.55 * skin) + skin_soft * (0.55 * skin)

    return np.clip(diffuse, 0.0, 1.0).astype(np.float32)


# ============================================================
# Person matte / masks
# ============================================================

def load_mask_float(path: str | Path, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw

    mask = Image.open(path).convert("L")
    mask = np.asarray(mask).astype(np.float32) / 255.0

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    return np.clip(mask, 0.0, 1.0)



def refine_loaded_person_mask(mask: np.ndarray) -> np.ndarray:
    """
    Prepare an externally generated soft alpha matte.

    Earlier versions converted every external mask into a binary mask,
    performed morphology, eroded it, and then re-feathered it. That was
    acceptable for coarse segmentation masks, but it destroys the soft
    hair and clothing-edge information produced by BiRefNet.

    This version preserves the original grayscale alpha values. It only:
      1. removes NaN/Inf values,
      2. applies an extremely small anti-aliasing blur,
      3. suppresses near-zero background noise smoothly,
      4. keeps confident foreground values close to their original value.
    """
    h, w = mask.shape

    alpha = np.nan_to_num(
        mask.astype(np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    # A tiny blur removes resize stair-stepping without widening the matte.
    alpha = blur01(alpha, scaled_kernel(3, h, w))

    # Smoothly remove tiny background probabilities while preserving
    # semi-transparent hair and anti-aliased boundaries.
    background_confidence = smoothstep(0.005, 0.060, alpha)
    alpha *= background_confidence

    # Keep highly confident foreground stable instead of pushing all
    # intermediate values through a hard threshold.
    foreground_confidence = smoothstep(0.90, 0.995, alpha)
    alpha = alpha * (1.0 - foreground_confidence) + foreground_confidence

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


def build_soft_matte_regions(
    person_alpha: np.ndarray,
    boundary_relight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a soft alpha matte into relighting regions.

    foreground_core:
        High-confidence subject interior. It can receive strong relighting.

    boundary_band:
        Hair, shoulders, clothes edges, and uncertain semi-transparent pixels.

    relight_matte:
        A conservative relighting envelope. Boundary pixels receive only a
        fraction of the interior relighting strength, while final compositing
        still uses the complete soft alpha matte.
    """
    person_alpha = np.clip(
        person_alpha.astype(np.float32),
        0.0,
        1.0,
    )

    foreground_core = smoothstep(
        0.52,
        0.90,
        person_alpha,
    )

    # Ensure the core never exceeds the source alpha.
    foreground_core = np.minimum(foreground_core, person_alpha)

    boundary_band = np.clip(
        person_alpha - foreground_core,
        0.0,
        1.0,
    )

    relight_matte = (
        foreground_core
        + float(boundary_relight) * boundary_band
    )
    relight_matte = np.minimum(relight_matte, person_alpha)

    return (
        foreground_core.astype(np.float32),
        boundary_band.astype(np.float32),
        np.clip(relight_matte, 0.0, 1.0).astype(np.float32),
    )

def build_skin_mask(image_srgb: np.ndarray) -> np.ndarray:
    h, w = image_srgb.shape[:2]

    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(image_u8, cv2.COLOR_RGB2YCrCb)

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    hsv_skin = (
        (hue >= 0) & (hue <= 32) &
        (sat >= 15) & (sat <= 225) &
        (val >= 30)
    )

    ycrcb_skin = (
        (cr >= 125) & (cr <= 198) &
        (cb >= 58) & (cb <= 165) &
        (y >= 22)
    )

    mask = (hsv_skin & ycrcb_skin).astype(np.float32)

    mask = blur01(mask, scaled_kernel(25, h, w))
    mask = smoothstep(0.08, 0.55, mask)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def center_prior(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0

    distance = np.sqrt((x / 0.95) ** 2 + ((y + 0.10) / 1.15) ** 2)
    prior = 1.0 - smoothstep(0.20, 1.15, distance)

    return np.clip(prior, 0.0, 1.0)


def build_fallback_person_matte(image_srgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Heuristic fallback only.

    For reliable results, pass a real person segmentation mask using --person-mask.
    """
    h, w = image_srgb.shape[:2]

    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)

    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = x_grid / max(w - 1, 1)
    y_norm = y_grid / max(h - 1, 1)

    cprior = center_prior(h, w)

    lower_body_prior = (
        smoothstep(0.00, 0.20, x_norm)
        * (1.0 - smoothstep(0.72, 0.98, x_norm))
        * smoothstep(0.32, 0.56, y_norm)
    )

    head_prior = (
        smoothstep(0.08, 0.28, x_norm)
        * (1.0 - smoothstep(0.62, 0.88, x_norm))
        * smoothstep(0.22, 0.38, y_norm)
        * (1.0 - smoothstep(0.66, 0.86, y_norm))
    )

    dark = 1.0 - smoothstep(0.18, 0.58, val)
    skin_core = smoothstep(0.12, 0.60, skin)

    skin_expand_u8 = (skin_core * 255.0).astype(np.uint8)
    dilate_k = scaled_size(25, h, w, min_value=9)

    skin_expand = cv2.dilate(
        skin_expand_u8,
        np.ones((dilate_k, dilate_k), np.uint8),
        iterations=1,
    )

    skin_blur_k = scaled_kernel(41, h, w)
    skin_expand = cv2.GaussianBlur(
        skin_expand,
        (skin_blur_k, skin_blur_k),
        0,
    ).astype(np.float32) / 255.0

    head_neighborhood = blur01(skin_core, scaled_kernel(95, h, w))
    hair_near_face = dark * head_neighborhood * (0.35 + 0.65 * head_prior)

    clothes = dark * lower_body_prior

    saturated_bg_reject = smoothstep(0.35, 0.85, sat) * (1.0 - skin_expand)

    broad = (
        0.30 * cprior +
        0.45 * lower_body_prior +
        0.25 * skin_expand
    )

    broad = blur01(broad, scaled_kernel(45, h, w))
    broad = smoothstep(0.28, 0.72, broad)

    candidate = np.maximum.reduce([
        0.95 * skin_core,
        0.90 * skin_expand,
        0.85 * clothes,
        0.80 * hair_near_face,
        0.45 * broad * (1.0 - 0.50 * saturated_bg_reject),
    ])

    top_reject = 1.0 - smoothstep(0.20, 0.34, y_norm)
    candidate *= 1.0 - 0.85 * top_reject * (1.0 - skin_expand) * (1.0 - hair_near_face)

    candidate = blur01(candidate, scaled_kernel(7, h, w))
    candidate = smoothstep(0.16, 0.48, candidate)

    binary = (candidate > 0.28).astype(np.uint8)

    close_k = scaled_size(17, h, w, min_value=5)
    open_k = scaled_size(5, h, w, min_value=3)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((close_k, close_k), np.uint8),
        iterations=1,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((open_k, open_k), np.uint8),
        iterations=1,
    )

    seed = np.maximum(skin_core, clothes)
    binary = keep_seeded_components(binary, seed)

    mask_u8 = (binary * 255).astype(np.uint8)

    erode_k = scaled_size(3, h, w, min_value=3)
    mask_u8 = cv2.erode(
        mask_u8,
        np.ones((erode_k, erode_k), np.uint8),
        iterations=1,
    )

    feather_k = scaled_kernel(23, h, w)
    matte = cv2.GaussianBlur(mask_u8, (feather_k, feather_k), 0).astype(np.float32) / 255.0

    matte = np.maximum(matte, 0.92 * skin_core)
    matte = blur01(matte, scaled_kernel(9, h, w))
    matte = smoothstep(0.03, 0.92, matte)

    return np.clip(matte, 0.0, 1.0).astype(np.float32)



def build_or_load_person_matte(
    image_srgb: np.ndarray,
    person_mask_path: str | None,
    boundary_relight: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    """
    Return:
        person_alpha
        foreground_core
        boundary_band
        relight_matte
        skin
        used_external_mask
    """
    h, w = image_srgb.shape[:2]

    skin = build_skin_mask(image_srgb)

    if person_mask_path is not None:
        raw_mask = load_mask_float(person_mask_path, (h, w))
        person_alpha = refine_loaded_person_mask(raw_mask)
        used_external_mask = True
    else:
        print(
            "Warning: no external soft matte supplied. "
            "Using the older heuristic person mask."
        )
        person_alpha = build_fallback_person_matte(image_srgb, skin)
        used_external_mask = False

    foreground_core, boundary_band, relight_matte = build_soft_matte_regions(
        person_alpha=person_alpha,
        boundary_relight=boundary_relight,
    )

    # Skin logic must never affect detected background pixels.
    skin = np.minimum(skin, person_alpha)

    return (
        person_alpha,
        foreground_core,
        boundary_band,
        relight_matte,
        skin,
        used_external_mask,
    )

def build_confident_person_matte(person_matte: np.ndarray) -> np.ndarray:
    h, w = person_matte.shape

    binary = (person_matte > 0.55).astype(np.uint8)

    erode_k = scaled_size(9, h, w, min_value=3)
    binary = cv2.erode(
        binary,
        np.ones((erode_k, erode_k), np.uint8),
        iterations=1,
    )

    blur_k = scaled_kernel(13, h, w)
    confident = cv2.GaussianBlur(
        (binary * 255).astype(np.uint8),
        (blur_k, blur_k),
        0,
    ).astype(np.float32) / 255.0

    confident = np.minimum(confident, person_matte)
    confident = smoothstep(0.08, 0.80, confident)

    return np.clip(confident, 0.0, 1.0).astype(np.float32)


# ============================================================
# Shadow / shading / albedo
# ============================================================

def build_shadow_mask(image_srgb: np.ndarray, image_linear: np.ndarray) -> np.ndarray:
    h, w = image_srgb.shape[:2]

    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)

    value = hsv[:, :, 2].astype(np.float32) / 255.0
    lab_l = lab[:, :, 0].astype(np.float32) / 255.0
    y = luminance(image_linear)

    tone = 0.45 * value + 0.35 * lab_l + 0.20 * np.sqrt(np.clip(y, 0.0, 1.0))

    small = blur(tone, scaled_kernel(7, h, w))
    mid = blur(tone, scaled_kernel(71, h, w))
    large = blur(tone, scaled_kernel(151, h, w))

    absolute_dark = 1.0 - smoothstep(0.14, 0.45, small)

    rel_mid = np.maximum(mid - small, 0.0) / np.maximum(mid, 0.08)
    rel_large = np.maximum(large - small, 0.0) / np.maximum(large, 0.08)

    local_shadow = np.maximum(
        smoothstep(0.05, 0.26, rel_mid),
        smoothstep(0.04, 0.22, rel_large),
    )

    shadow = np.maximum(0.45 * absolute_dark, local_shadow)

    bright_reject = smoothstep(0.60, 0.84, small)
    shadow *= 1.0 - 0.70 * bright_reject

    shadow = blur01(shadow, scaled_kernel(31, h, w))
    shadow = smoothstep(0.12, 0.82, shadow)

    return np.clip(shadow, 0.0, 1.0).astype(np.float32)


def compute_total_shading(diffuse: np.ndarray, ambient: float) -> np.ndarray:
    return np.clip(ambient + (1.0 - ambient) * diffuse, 0.02, 1.0).astype(np.float32)


def estimate_albedo_inside_person(
    image_linear: np.ndarray,
    old_total: np.ndarray,
    shadow_person: np.ndarray,
    confident_matte: np.ndarray,
    person_matte: np.ndarray,
    albedo_floor: float,
) -> np.ndarray:
    floor = albedo_floor + 0.22 * shadow_person
    denom = np.maximum(old_total, floor)

    albedo = image_linear / denom[:, :, None]
    albedo = np.nan_to_num(albedo, nan=0.0, posinf=1.4, neginf=0.0)

    stats_mask = confident_matte > 0.45

    if np.count_nonzero(stats_mask) < 200:
        stats_mask = person_matte > 0.20

    if np.count_nonzero(stats_mask) < 200:
        stats_mask = np.ones(old_total.shape, dtype=bool)

    for c in range(3):
        values = albedo[:, :, c][stats_mask]

        hi = np.percentile(values, 99.2)
        lo = np.percentile(values, 0.2)

        albedo[:, :, c] = np.clip(
            albedo[:, :, c],
            max(0.0, lo),
            max(hi, 0.6),
        )

    albedo = np.clip(albedo, 0.0, 1.4)

    return albedo.astype(np.float32)


def blinn_specular(normal: np.ndarray, light: np.ndarray, shininess: float) -> np.ndarray:
    light = normalize_vector(light)

    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = normalize_vector(light + view)

    ndoth = np.sum(normal * half_vec.reshape(1, 1, 3), axis=2)
    ndoth = np.clip(ndoth, 0.0, 1.0)

    spec = np.power(ndoth, shininess).astype(np.float32)

    h, w = spec.shape
    spec = blur01(spec, scaled_kernel(17, h, w))

    return spec


def preserve_luma_safety(
    relit: np.ndarray,
    original: np.ndarray,
    person_matte: np.ndarray,
    max_darken_amount: float,
) -> np.ndarray:
    old_y = luminance(original)
    new_y = luminance(relit)

    min_y = old_y * (1.0 - max_darken_amount)

    scale = np.where(
        new_y < min_y,
        min_y / np.maximum(new_y, 1e-5),
        1.0,
    )

    scale = np.clip(scale, 1.0, 1.6)

    protected = np.clip(relit * scale[:, :, None], 0.0, 1.0)

    matte_3c = person_matte[:, :, None]

    return np.clip(
        original * (1.0 - matte_3c) + protected * matte_3c,
        0.0,
        1.0,
    )


def composite_original_background(
    original_linear: np.ndarray,
    relit_linear: np.ndarray,
    person_matte: np.ndarray,
) -> np.ndarray:
    matte_3c = np.clip(person_matte, 0.0, 1.0)[:, :, None]

    return np.clip(
        original_linear * (1.0 - matte_3c) + relit_linear * matte_3c,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# Optional old-light estimation
# ============================================================

def auto_estimate_old_light(
    image_linear: np.ndarray,
    normal: np.ndarray,
    confident_matte: np.ndarray,
) -> np.ndarray:
    mask = confident_matte > 0.45

    if np.count_nonzero(mask) < 300:
        print("Auto old-light skipped: confident person matte too small.")
        return np.array([-0.35, -0.25, 0.90], dtype=np.float32)

    y = luminance(image_linear)
    h, w = y.shape

    y_blur = blur(y, scaled_kernel(31, h, w))

    y_values = y_blur[mask]
    y_values = y_values - np.mean(y_values)
    y_std = np.std(y_values)

    if y_std < 1e-6:
        print("Auto old-light skipped: luminance variation too small.")
        return np.array([-0.35, -0.25, 0.90], dtype=np.float32)

    best_score = -1e9
    best_light = np.array([-0.35, -0.25, 0.90], dtype=np.float32)

    xy_values = np.linspace(-0.85, 0.85, 11, dtype=np.float32)
    z_values = np.array([0.45, 0.60, 0.75, 0.90, 1.10], dtype=np.float32)

    for x in xy_values:
        for yy in xy_values:
            for z in z_values:
                candidate = normalize_vector(np.array([x, yy, z], dtype=np.float32))

                shade = lambert(normal, candidate)
                shade = smooth_shading(shade)

                s_values = shade[mask]
                s_values = s_values - np.mean(s_values)
                s_std = np.std(s_values)

                if s_std < 1e-6:
                    continue

                score = float(np.mean((s_values / s_std) * (y_values / y_std)))

                if score > best_score:
                    best_score = score
                    best_light = candidate

    print(
        "Auto old-light selected:",
        f"[{best_light[0]:.3f}, {best_light[1]:.3f}, {best_light[2]:.3f}]",
        f"correlation={best_score:.3f}",
    )

    return best_light.astype(np.float32)


# ============================================================
# Main relighting function
# ============================================================


def relight_person_only(
    image_srgb: np.ndarray,
    normal: np.ndarray,
    person_mask_path: str | None,
    old_light: np.ndarray,
    new_light: np.ndarray,
    ambient_old: float,
    ambient_new: float,
    strength: float,
    boundary_relight: float,
    background_strength: float,
    shadow_relight: float,
    albedo_floor: float,
    specular_strength: float,
    specular_shininess: float,
    max_darken_amount: float,
    exposure: float,
    auto_old_light_enabled: bool,
    background_lock: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image_linear = srgb_to_linear(image_srgb)
    h, w = image_srgb.shape[:2]

    (
        person_alpha,
        foreground_core,
        boundary_band,
        relight_matte,
        skin,
        used_external_mask,
    ) = build_or_load_person_matte(
        image_srgb=image_srgb,
        person_mask_path=person_mask_path,
        boundary_relight=boundary_relight,
    )

    # This stricter interior is used only for lighting/albedo statistics.
    # Final compositing always retains the complete BiRefNet alpha matte.
    confident_matte = build_confident_person_matte(foreground_core)

    shadow = build_shadow_mask(image_srgb, image_linear)
    shadow_person = shadow * person_alpha

    normal_edge = build_normal_edge_mask(normal)

    if auto_old_light_enabled:
        old_light = auto_estimate_old_light(
            image_linear=image_linear,
            normal=normal,
            confident_matte=confident_matte,
        )

    old_diffuse = compute_smoothed_diffuse(
        normal=normal,
        light=old_light,
        normal_edge=normal_edge,
        skin=skin,
    )

    new_diffuse = compute_smoothed_diffuse(
        normal=normal,
        light=new_light,
        normal_edge=normal_edge,
        skin=skin,
    )

    old_total = compute_total_shading(old_diffuse, ambient_old)
    new_total = compute_total_shading(new_diffuse, ambient_new)

    albedo = estimate_albedo_inside_person(
        image_linear=image_linear,
        old_total=old_total,
        shadow_person=shadow_person,
        confident_matte=confident_matte,
        person_matte=person_alpha,
        albedo_floor=albedo_floor,
    )

    physical = albedo * new_total[:, :, None]
    physical = np.clip(physical, 0.0, 1.0)

    spec = blinn_specular(
        normal=normal,
        light=new_light,
        shininess=specular_shininess,
    )

    spec_mask = confident_matte.copy()
    spec_mask *= 1.0 - 0.75 * shadow_person
    spec_mask *= 1.0 - 0.55 * normal_edge
    spec_mask *= 1.0 - 0.40 * skin
    spec_mask = blur01(spec_mask, scaled_kernel(21, h, w))
    spec_mask = np.minimum(spec_mask, relight_matte)

    physical += specular_strength * spec[:, :, None] * spec_mask[:, :, None]
    physical = np.clip(physical, 0.0, 1.0)

    background = 1.0 - person_alpha

    # Full relighting in the confident subject interior; reduced relighting
    # through hair, antialiased contours, and uncertain matte boundaries.
    relight_strength = relight_matte * strength

    # Retained only for backward compatibility and debugging. Keep this at
    # zero in normal use. Strict background locking later restores the scene.
    relight_strength += background * background_strength

    relight_strength *= 1.0 - (1.0 - shadow_relight) * shadow_person
    relight_strength *= 1.0 - 0.55 * normal_edge
    relight_strength *= 1.0 - 0.18 * skin

    relight_strength = blur01(
        relight_strength,
        scaled_kernel(23, h, w),
    )

    # Blur can spread relighting beyond the original region. Clamp rather
    # than multiply so that soft edges are not squared and made too thin.
    if background_strength <= 1e-8:
        relight_strength = np.minimum(
            relight_strength,
            relight_matte,
        )
    else:
        maximum_allowed = np.clip(
            relight_matte + background * background_strength,
            0.0,
            1.0,
        )
        relight_strength = np.minimum(
            relight_strength,
            maximum_allowed,
        )

    relight_strength = np.clip(relight_strength, 0.0, 1.0)

    relit_linear = (
        image_linear * (1.0 - relight_strength[:, :, None])
        + physical * relight_strength[:, :, None]
    )

    relit_linear = preserve_luma_safety(
        relit=relit_linear,
        original=image_linear,
        person_matte=person_alpha,
        max_darken_amount=max_darken_amount,
    )

    relit_linear = np.clip(relit_linear * exposure, 0.0, 1.0)

    if background_lock:
        final_linear = composite_original_background(
            original_linear=image_linear,
            relit_linear=relit_linear,
            person_matte=person_alpha,
        )
    else:
        final_linear = relit_linear

    output_srgb = linear_to_srgb(final_linear)
    output_srgb = np.clip(output_srgb, 0.0, 1.0)

    albedo_luma = luminance(np.clip(albedo, 0.0, 1.0))

    debug = {
        # Keep the previous filename for compatibility.
        "person_matte_FINAL_BLEND": person_alpha,
        "person_alpha_BIREFNET": person_alpha,
        "foreground_core": foreground_core,
        "boundary_band": boundary_band,
        "boundary_relight_envelope": relight_matte,
        "confident_person_matte": confident_matte,
        "background_restored_area": 1.0 - person_alpha,
        "skin": skin,
        "shadow": shadow,
        "shadow_person": shadow_person,
        "normal_edge": normal_edge,
        "old_diffuse": old_diffuse,
        "new_diffuse": new_diffuse,
        "old_total": old_total,
        "new_total": new_total,
        "albedo_luma": albedo_luma,
        "relight_strength": relight_strength,
        "specular": spec * spec_mask,
        "used_external_mask": np.full(
            (h, w),
            1.0 if used_external_mask else 0.0,
            dtype=np.float32,
        ),
    }

    return output_srgb, debug

# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sunit Stage 5 V8: BiRefNet soft-matte-aware person relighting."
    )

    parser.add_argument("--image", required=True, help="Input RGB image.")
    parser.add_argument("--normal", required=True, help="DSINE/normal map image.")
    parser.add_argument("--output", required=True, help="Output relit image.")

    parser.add_argument(
        "--person-mask",
        default=None,
        help=(
            "Soft person alpha matte. Grayscale values are preserved. "
            "BiRefNet output is recommended."
        ),
    )

    parser.add_argument(
        "--matte",
        default=None,
        help="Alias for --person-mask. Intended for BiRefNet soft alpha output.",
    )

    parser.add_argument(
        "--foreground-mask",
        default=None,
        help="Backward-compatible alias for --person-mask.",
    )

    parser.add_argument(
        "--old-light",
        nargs=3,
        type=float,
        default=[-0.35, -0.25, 0.90],
        help="Old/original light direction x y z.",
    )

    parser.add_argument(
        "--new-light",
        nargs=3,
        type=float,
        default=[0.70, -0.30, 0.65],
        help="New light direction x y z.",
    )

    parser.add_argument(
        "--auto-old-light",
        action="store_true",
        help="Estimate old light from image luminance and normals inside person matte.",
    )

    parser.add_argument("--ambient-old", type=float, default=0.45)
    parser.add_argument("--ambient-new", type=float, default=0.35)

    parser.add_argument(
        "--ambient",
        type=float,
        default=None,
        help="Alias for --ambient-new for compatibility with older commands.",
    )

    parser.add_argument("--strength", type=float, default=None)

    parser.add_argument(
        "--person-strength",
        type=float,
        default=None,
        help="Alias for --strength.",
    )

    parser.add_argument(
        "--boundary-relight",
        type=float,
        default=0.35,
        help=(
            "Fraction of normal relighting applied to soft hair/clothing "
            "boundaries. Recommended range: 0.20 to 0.50."
        ),
    )

    parser.add_argument(
        "--background-strength",
        type=float,
        default=0.00,
        help="Keep this 0.00 for clean background protection.",
    )

    parser.add_argument(
        "--shadow-relight",
        type=float,
        default=0.45,
        help="0 locks old shadows, 1 fully relights shadows. Good range: 0.35 to 0.65.",
    )

    parser.add_argument(
        "--albedo-floor",
        type=float,
        default=0.28,
        help="Higher prevents shadow/albedo explosion. Good range: 0.22 to 0.36.",
    )

    parser.add_argument("--specular", type=float, default=None)

    parser.add_argument(
        "--highlight",
        type=float,
        default=None,
        help="Alias for --specular.",
    )

    parser.add_argument("--specular-shininess", type=float, default=72.0)

    parser.add_argument(
        "--max-darken",
        type=float,
        default=0.18,
        help="Maximum allowed luminance darkening compared to original.",
    )

    parser.add_argument(
        "--exposure",
        type=float,
        default=1.0,
        help="Person exposure multiplier. Background is restored afterward.",
    )

    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")

    parser.add_argument(
        "--disable-background-lock",
        action="store_true",
        help="Debug only. Allows the background to be changed.",
    )

    parser.add_argument("--save-debug", action="store_true")

    return parser.parse_args()


def resolve_compat_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.strength is not None:
        args.final_strength = args.strength
    elif args.person_strength is not None:
        args.final_strength = args.person_strength
    else:
        args.final_strength = 0.62

    if args.ambient is not None:
        args.final_ambient_new = args.ambient
        args.final_ambient_old = max(args.ambient_old, args.ambient + 0.05)
    else:
        args.final_ambient_old = args.ambient_old
        args.final_ambient_new = args.ambient_new

    if args.specular is not None:
        args.final_specular = args.specular
    elif args.highlight is not None:
        args.final_specular = args.highlight
    else:
        args.final_specular = 0.035

    if args.person_mask is not None:
        args.final_person_mask = args.person_mask
    elif args.matte is not None:
        args.final_person_mask = args.matte
    else:
        args.final_person_mask = args.foreground_mask

    return args


def validate_args(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    normal_path = Path(args.normal)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not normal_path.exists():
        raise FileNotFoundError(f"Normal map not found: {normal_path}")

    if args.final_person_mask is not None and not Path(args.final_person_mask).exists():
        raise FileNotFoundError(f"Person mask not found: {args.final_person_mask}")

    checks = [
        ("ambient_old", args.final_ambient_old, 0.0, 1.0),
        ("ambient_new", args.final_ambient_new, 0.0, 1.0),
        ("strength/person-strength", args.final_strength, 0.0, 1.0),
        ("boundary_relight", args.boundary_relight, 0.0, 1.0),
        ("background_strength", args.background_strength, 0.0, 1.0),
        ("shadow_relight", args.shadow_relight, 0.0, 1.0),
        ("albedo_floor", args.albedo_floor, 0.05, 0.80),
        ("max_darken", args.max_darken, 0.0, 0.80),
        ("exposure", args.exposure, 0.50, 1.50),
    ]

    for name, value, low, high in checks:
        if value < low or value > high:
            raise ValueError(f"{name} must be between {low} and {high}. Got {value}.")

    if args.final_specular < 0.0 or args.final_specular > 0.30:
        raise ValueError(
            f"specular/highlight should be between 0.0 and 0.30. Got {args.final_specular}."
        )

    if args.specular_shininess < 1.0:
        raise ValueError("specular-shininess must be >= 1.0.")

    normalize_vector(np.array(args.old_light, dtype=np.float32))
    normalize_vector(np.array(args.new_light, dtype=np.float32))


def save_debug_maps(output_path: str | Path, debug: dict[str, np.ndarray]) -> None:
    output_path = Path(output_path)
    debug_dir = output_path.parent / f"{output_path.stem}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for name, value in debug.items():
        save_gray(debug_dir / f"{name}.png", value)

    print(f"Saved debug maps to: {debug_dir}")
    print("Inspect: person_alpha_BIREFNET.png")
    print("Inspect: boundary_band.png and relight_strength.png")
    print("The background should remain black in relight_strength.png.")


def main() -> None:
    args = resolve_compat_args(parse_args())
    validate_args(args)

    image_srgb = load_rgb(args.image)
    h, w = image_srgb.shape[:2]

    normal = load_normal_map(
        path=args.normal,
        target_hw=(h, w),
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
    )

    output, debug = relight_person_only(
        image_srgb=image_srgb,
        normal=normal,
        person_mask_path=args.final_person_mask,
        old_light=np.array(args.old_light, dtype=np.float32),
        new_light=np.array(args.new_light, dtype=np.float32),
        ambient_old=args.final_ambient_old,
        ambient_new=args.final_ambient_new,
        strength=args.final_strength,
        boundary_relight=args.boundary_relight,
        background_strength=args.background_strength,
        shadow_relight=args.shadow_relight,
        albedo_floor=args.albedo_floor,
        specular_strength=args.final_specular,
        specular_shininess=args.specular_shininess,
        max_darken_amount=args.max_darken,
        exposure=args.exposure,
        auto_old_light_enabled=args.auto_old_light,
        background_lock=not args.disable_background_lock,
    )

    save_rgb(args.output, output)
    print(f"Saved Stage 5 V8 output to: {args.output}")

    if args.save_debug:
        save_debug_maps(args.output, debug)


if __name__ == "__main__":
    main()
