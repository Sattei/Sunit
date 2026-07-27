"""Archived V5 experiment; superseded by the canonical V8 engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ============================================================
# IO / Color
# ============================================================

def load_rgb(path: str) -> np.ndarray:
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

    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)

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
    value = max(3, int(value))
    if value % 2 == 0:
        value += 1
    return value


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


# ============================================================
# Normal map
# ============================================================

def load_normal_map(
    path: str,
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
    smooth = blur(shading, 35)

    smooth = cv2.bilateralFilter(
        smooth.astype(np.float32),
        d=11,
        sigmaColor=0.08,
        sigmaSpace=35,
    )

    return np.clip(smooth, 0.0, 1.0)


def blend_raw_and_smooth_shading(
    raw: np.ndarray,
    normal_edge: np.ndarray,
    skin: np.ndarray,
) -> np.ndarray:
    smooth = smooth_shading(raw)

    smooth_weight = np.clip(
        0.20 + 0.45 * skin + 0.60 * normal_edge,
        0.0,
        0.90,
    )

    mixed = raw * (1.0 - smooth_weight) + smooth * smooth_weight
    return np.clip(mixed, 0.0, 1.0)


# ============================================================
# Masks
# ============================================================

def center_prior(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)

    x = (x / max(width - 1, 1)) * 2.0 - 1.0
    y = (y / max(height - 1, 1)) * 2.0 - 1.0

    distance = np.sqrt((x / 0.95) ** 2 + ((y + 0.10) / 1.15) ** 2)
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
        (s >= 18) & (s <= 220) &
        (v >= 35)
    )

    ycrcb_skin = (
        (cr >= 128) & (cr <= 194) &
        (cb >= 60) & (cb <= 160) &
        (y >= 25)
    )

    mask = (hsv_skin & ycrcb_skin).astype(np.float32)
    mask = blur01(mask, 25)
    mask = smoothstep(0.08, 0.55, mask)

    return np.clip(mask, 0.0, 1.0)


def build_foreground_mask(image_srgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Broad foreground estimate used internally.

    This is intentionally not trusted for final compositing anymore, because
    it can include door/wall/background regions. Final background protection
    uses build_safe_person_matte().
    """
    h, w = image_srgb.shape[:2]

    cprior = center_prior(h, w)

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = x / max(w - 1, 1)
    y_norm = y / max(h - 1, 1)

    body_prior = (
        smoothstep(0.02, 0.22, x_norm)
        * (1.0 - smoothstep(0.76, 0.98, x_norm))
        * smoothstep(0.30, 0.55, y_norm)
    )

    head_skin_prior = blur01(skin, 71)

    foreground = (
        0.26 * cprior +
        0.44 * body_prior +
        0.30 * head_skin_prior
    )

    foreground = blur01(foreground, 45)
    foreground = smoothstep(0.28, 0.72, foreground)

    return np.clip(foreground, 0.0, 1.0)


def load_optional_mask(path: str | None, target_hw: tuple[int, int]) -> np.ndarray | None:
    if path is None:
        return None

    h, w = target_hw

    mask = Image.open(path).convert("L")
    mask = np.asarray(mask).astype(np.float32) / 255.0

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = blur01(mask, 25)
    mask = smoothstep(0.08, 0.85, mask)

    return np.clip(mask, 0.0, 1.0)


def build_shadow_mask(image_srgb: np.ndarray, image_linear: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)

    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB)

    value = hsv[:, :, 2].astype(np.float32) / 255.0
    lab_l = lab[:, :, 0].astype(np.float32) / 255.0
    y = luminance(image_linear)

    tone = 0.45 * value + 0.35 * lab_l + 0.20 * np.sqrt(np.clip(y, 0.0, 1.0))

    small = blur(tone, 7)
    mid = blur(tone, 71)
    large = blur(tone, 151)

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

    shadow = blur01(shadow, 31)
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
    edge = blur01(edge, 17)

    return np.clip(edge, 0.0, 1.0)


def keep_seeded_components(binary: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """
    Keep only connected components that overlap likely person pixels.
    This helps remove background islands such as door closers, gate edges,
    signs, and wall patches.
    """
    binary_u8 = binary.astype(np.uint8)
    seed_bool = seed > 0.15

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, 8)

    if num_labels <= 1:
        return binary.astype(np.float32)

    h, w = binary.shape
    total_area = h * w
    kept = np.zeros_like(binary_u8)

    largest_label = 0
    largest_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > largest_area:
            largest_area = area
            largest_label = label

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(40, int(total_area * 0.0004)):
            continue

        component = labels == label
        seed_overlap = int(np.count_nonzero(component & seed_bool))

        if seed_overlap > max(12, area * 0.015):
            kept[component] = 1

    if np.count_nonzero(kept) == 0 and largest_label > 0:
        kept[labels == largest_label] = 1

    return kept.astype(np.float32)


def build_safe_person_matte(
    image_srgb: np.ndarray,
    broad_foreground: np.ndarray,
    skin: np.ndarray,
) -> np.ndarray:
    """
    Tight person matte for the final composite.

    This is the important Stage 5 fix:
    - the broad foreground mask may include the door/gate/background;
    - this matte is stricter and mostly keeps skin, hair, arm, and torso;
    - the final output copies original background back using this matte.
    """
    h, w = image_srgb.shape[:2]

    image_u8 = (np.clip(image_srgb, 0.0, 1.0) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)

    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0

    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = x_grid / max(w - 1, 1)
    y_norm = y_grid / max(h - 1, 1)

    lower_body_prior = (
        smoothstep(0.00, 0.18, x_norm)
        * (1.0 - smoothstep(0.70, 0.95, x_norm))
        * smoothstep(0.34, 0.56, y_norm)
    )

    head_prior = (
        smoothstep(0.08, 0.26, x_norm)
        * (1.0 - smoothstep(0.62, 0.86, x_norm))
        * smoothstep(0.25, 0.38, y_norm)
        * (1.0 - smoothstep(0.64, 0.82, y_norm))
    )

    person_position_prior = np.maximum(lower_body_prior, head_prior)

    dark = 1.0 - smoothstep(0.18, 0.55, val)
    dark = np.clip(dark, 0.0, 1.0)

    skin_core = smoothstep(0.12, 0.60, skin)

    skin_u8 = (np.clip(skin_core, 0.0, 1.0) * 255).astype(np.uint8)
    skin_expand = cv2.dilate(skin_u8, np.ones((25, 25), np.uint8), iterations=1)
    skin_expand = cv2.GaussianBlur(skin_expand, (41, 41), 0).astype(np.float32) / 255.0
    skin_expand = np.clip(skin_expand, 0.0, 1.0)

    head_neighborhood = blur01(skin_core, 95)
    hair_near_face = dark * head_neighborhood * (0.35 + 0.65 * head_prior)

    clothes = dark * lower_body_prior

    saturated_bg_reject = smoothstep(0.35, 0.85, sat) * (1.0 - skin_expand)
    broad_limited = broad_foreground * person_position_prior * (1.0 - 0.50 * saturated_bg_reject)

    candidate = np.maximum.reduce([
        0.95 * skin_core,
        0.95 * skin_expand,
        0.85 * clothes,
        0.80 * hair_near_face,
        0.45 * broad_limited,
    ])

    top_reject = 1.0 - smoothstep(0.20, 0.34, y_norm)
    candidate *= 1.0 - 0.85 * top_reject * (1.0 - skin_expand) * (1.0 - hair_near_face)

    candidate = blur01(candidate, 7)
    candidate = smoothstep(0.16, 0.48, candidate)

    binary = (candidate > 0.28).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    seed = np.maximum(skin_core, clothes)
    binary = keep_seeded_components(binary, seed)

    matte_u8 = (binary * 255).astype(np.uint8)
    matte_u8 = cv2.erode(matte_u8, np.ones((3, 3), np.uint8), iterations=1)
    matte_u8 = cv2.GaussianBlur(matte_u8, (23, 23), 0)
    matte = matte_u8.astype(np.float32) / 255.0

    matte = np.maximum(matte, 0.92 * skin_core)

    matte = blur01(matte, 9)
    matte = smoothstep(0.03, 0.92, matte)

    return np.clip(matte, 0.0, 1.0).astype(np.float32)


# ============================================================
# Physical / intrinsic relighting
# ============================================================

def compute_total_shading(
    diffuse: np.ndarray,
    ambient: float,
) -> np.ndarray:
    return np.clip(ambient + (1.0 - ambient) * diffuse, 0.02, 1.0)


def estimate_albedo(
    image_linear: np.ndarray,
    old_total: np.ndarray,
    shadow: np.ndarray,
    albedo_floor: float,
) -> np.ndarray:
    """
    Estimate reflectance/albedo.

    We do not divide by very tiny old shading because that explodes shadows.
    In shadow regions the floor is raised automatically.
    """
    floor = albedo_floor + 0.22 * shadow
    denom = np.maximum(old_total, floor)

    albedo = image_linear / denom[:, :, None]

    for c in range(3):
        hi = np.percentile(albedo[:, :, c], 99.2)
        albedo[:, :, c] = np.clip(albedo[:, :, c], 0.0, max(hi, 0.6))

    albedo = np.clip(albedo, 0.0, 1.4)

    return albedo.astype(np.float32)


def blinn_specular(
    normal: np.ndarray,
    light: np.ndarray,
    shininess: float,
) -> np.ndarray:
    light = normalize_vector(light)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    half_vec = normalize_vector(light + view)

    ndoth = np.sum(normal * half_vec.reshape(1, 1, 3), axis=2)
    ndoth = np.clip(ndoth, 0.0, 1.0)

    spec = np.power(ndoth, shininess)
    spec = blur01(spec, 17)

    return spec


def preserve_luma_safety(
    relit: np.ndarray,
    original: np.ndarray,
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

    return np.clip(relit * scale[:, :, None], 0.0, 1.0)


def composite_original_background(
    original_linear: np.ndarray,
    relit_linear: np.ndarray,
    person_matte: np.ndarray,
) -> np.ndarray:
    """
    Hard background protection.

    This is the final lock:
    background pixels are copied from the original image after all relighting,
    so global exposure/specular/shadow operations cannot damage the door/gate.
    """
    matte_3c = np.clip(person_matte, 0.0, 1.0)[:, :, None]
    return np.clip(
        original_linear * (1.0 - matte_3c) + relit_linear * matte_3c,
        0.0,
        1.0,
    )


def stage5_v6_person_locked_relight(
    image_srgb: np.ndarray,
    normal: np.ndarray,
    foreground_mask: np.ndarray | None,
    old_light: np.ndarray,
    new_light: np.ndarray,
    ambient_old: float,
    ambient_new: float,
    strength: float,
    background_strength: float,
    shadow_relight: float,
    albedo_floor: float,
    specular_strength: float,
    specular_shininess: float,
    max_darken_amount: float,
    exposure: float,
    background_lock: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    image_linear = srgb_to_linear(image_srgb)

    skin = build_skin_mask(image_srgb)

    if foreground_mask is None:
        broad_foreground = build_foreground_mask(image_srgb, skin)
    else:
        broad_foreground = foreground_mask

    person_matte = build_safe_person_matte(
        image_srgb=image_srgb,
        broad_foreground=broad_foreground,
        skin=skin,
    )

    background = blur01(1.0 - person_matte, 35)

    shadow = build_shadow_mask(image_srgb, image_linear)
    normal_edge = build_normal_edge_mask(normal)

    old_raw = lambert(normal, old_light)
    new_raw = lambert(normal, new_light)

    old_diffuse = blend_raw_and_smooth_shading(
        raw=old_raw,
        normal_edge=normal_edge,
        skin=skin,
    )

    new_diffuse = blend_raw_and_smooth_shading(
        raw=new_raw,
        normal_edge=normal_edge,
        skin=skin,
    )

    old_skin_soft = smooth_shading(old_diffuse)
    new_skin_soft = smooth_shading(new_diffuse)

    old_diffuse = old_diffuse * (1.0 - 0.55 * skin) + old_skin_soft * (0.55 * skin)
    new_diffuse = new_diffuse * (1.0 - 0.55 * skin) + new_skin_soft * (0.55 * skin)

    old_total = compute_total_shading(old_diffuse, ambient_old)
    new_total = compute_total_shading(new_diffuse, ambient_new)

    albedo = estimate_albedo(
        image_linear=image_linear,
        old_total=old_total,
        shadow=shadow,
        albedo_floor=albedo_floor,
    )

    physical = albedo * new_total[:, :, None]
    physical = np.clip(physical, 0.0, 1.0)

    spec = blinn_specular(
        normal=normal,
        light=new_light,
        shininess=specular_shininess,
    )

    spec_mask = person_matte.copy()
    spec_mask *= 1.0 - 0.75 * shadow
    spec_mask *= 1.0 - 0.55 * normal_edge
    spec_mask *= 1.0 - 0.45 * skin
    spec_mask = blur01(spec_mask, 21)

    physical += specular_strength * spec[:, :, None] * spec_mask[:, :, None]
    physical = np.clip(physical, 0.0, 1.0)

    relight_strength = person_matte * strength

    relight_strength += background * background_strength

    relight_strength *= 1.0 - (1.0 - shadow_relight) * shadow
    relight_strength *= 1.0 - 0.55 * normal_edge
    relight_strength *= 1.0 - 0.18 * skin

    relight_strength = blur01(relight_strength, 23)
    relight_strength = np.clip(relight_strength, 0.0, 1.0)

    relit_linear = (
        image_linear * (1.0 - relight_strength[:, :, None]) +
        physical * relight_strength[:, :, None]
    )

    relit_linear = preserve_luma_safety(
        relit=relit_linear,
        original=image_linear,
        max_darken_amount=max_darken_amount,
    )

    relit_linear = np.clip(relit_linear * exposure, 0.0, 1.0)

    if background_lock:
        final_linear = composite_original_background(
            original_linear=image_linear,
            relit_linear=relit_linear,
            person_matte=person_matte,
        )
    else:
        final_linear = relit_linear

    output_srgb = linear_to_srgb(final_linear)
    output_srgb = np.clip(output_srgb, 0.0, 1.0)

    debug = {
        "broad_foreground": broad_foreground,
        "person_matte_FINAL_BLEND": person_matte,
        "background_from_person_matte": background,
        "skin": skin,
        "shadow": shadow,
        "normal_edge": normal_edge,
        "old_raw": old_raw,
        "new_raw": new_raw,
        "old_diffuse": old_diffuse,
        "new_diffuse": new_diffuse,
        "old_total": old_total,
        "new_total": new_total,
        "albedo_luma": luminance(np.clip(albedo, 0.0, 1.0)),
        "relight_strength": relight_strength,
        "specular": spec * spec_mask,
    }

    return output_srgb, debug


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Sunit Stage 5 V6: person-locked relighting with hard background restore."
    )

    parser.add_argument("--image", required=True)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--foreground-mask", default=None)

    parser.add_argument(
        "--old-light",
        nargs=3,
        type=float,
        default=[-0.35, -0.25, 0.90],
    )

    parser.add_argument(
        "--new-light",
        nargs=3,
        type=float,
        default=[0.70, -0.30, 0.65],
    )

    parser.add_argument("--ambient-old", type=float, default=0.45)
    parser.add_argument("--ambient-new", type=float, default=0.35)

    parser.add_argument(
        "--ambient",
        type=float,
        default=None,
        help="Alias for --ambient-new. Useful for older commands.",
    )

    parser.add_argument("--strength", type=float, default=None)

    parser.add_argument(
        "--person-strength",
        type=float,
        default=None,
        help="Alias for --strength. Useful for older commands.",
    )

    parser.add_argument(
        "--background-strength",
        type=float,
        default=0.00,
        help="Keep 0.00 while background lock is enabled.",
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
        help="Higher value prevents shadow explosion. Good range: 0.22 to 0.36.",
    )

    parser.add_argument("--specular", type=float, default=None)

    parser.add_argument(
        "--highlight",
        type=float,
        default=None,
        help="Alias for --specular. Useful for older commands.",
    )

    parser.add_argument("--specular-shininess", type=float, default=72.0)

    parser.add_argument(
        "--max-darken",
        type=float,
        default=0.18,
        help="Maximum allowed luminance darkening compared to original.",
    )

    parser.add_argument("--exposure", type=float, default=1.0)

    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--flip-z", action="store_true")

    parser.add_argument(
        "--disable-background-lock",
        action="store_true",
        help="Debug only. Allows the background to be changed by the relighting pipeline.",
    )

    parser.add_argument("--save-debug", action="store_true")

    return parser.parse_args()


def resolve_args(args):
    if args.strength is not None:
        strength = args.strength
    elif args.person_strength is not None:
        strength = args.person_strength
    else:
        strength = 0.62

    if args.ambient is not None:
        ambient_new = args.ambient
        ambient_old = max(args.ambient_old, args.ambient + 0.05)
    else:
        ambient_old = args.ambient_old
        ambient_new = args.ambient_new

    if args.specular is not None:
        specular = args.specular
    elif args.highlight is not None:
        specular = args.highlight
    else:
        specular = 0.035

    return strength, ambient_old, ambient_new, specular


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

    strength, ambient_old, ambient_new, specular = resolve_args(args)

    output, debug = stage5_v6_person_locked_relight(
        image_srgb=image_srgb,
        normal=normal,
        foreground_mask=foreground_mask,
        old_light=np.array(args.old_light, dtype=np.float32),
        new_light=np.array(args.new_light, dtype=np.float32),
        ambient_old=ambient_old,
        ambient_new=ambient_new,
        strength=strength,
        background_strength=args.background_strength,
        shadow_relight=args.shadow_relight,
        albedo_floor=args.albedo_floor,
        specular_strength=specular,
        specular_shininess=args.specular_shininess,
        max_darken_amount=args.max_darken,
        exposure=args.exposure,
        background_lock=not args.disable_background_lock,
    )

    save_rgb(args.output, output)
    print(f"Saved Stage 5 V6 output to: {args.output}")

    if args.save_debug:
        output_path = Path(args.output)
        debug_dir = output_path.parent / f"{output_path.stem}_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for name, value in debug.items():
            save_gray(debug_dir / f"{name}.png", value)

        print(f"Saved debug maps to: {debug_dir}")
        print("Check person_matte_FINAL_BLEND.png. White = relit person, black = untouched original background.")


if __name__ == "__main__":
    main()
