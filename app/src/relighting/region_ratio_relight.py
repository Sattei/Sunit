import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image = np.clip(image, 0.0, 1.0)
    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(image_u8, mode="RGB").save(path)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    """
    Save a single-channel debug image in [0, 1].
    """
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
        1.055 * (image ** (1.0 / 2.4)) - 0.055,
    )


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)

    if length < 1e-8:
        raise ValueError("Light direction cannot be [0, 0, 0].")

    return vector / length


def read_normal_map(
    path: str | Path,
    target_size: tuple[int, int],
    flip_y: bool,
) -> np.ndarray:
    """
    Read normal map encoded as RGB.

    Expected encoding:
        RGB [0, 255] -> normal [-1, 1]
    """
    width, height = target_size

    normal_image = Image.open(path).convert("RGB")

    if normal_image.size != target_size:
        normal_image = normal_image.resize(target_size, Image.Resampling.BILINEAR)

    normals = np.asarray(normal_image).astype(np.float32) / 255.0
    normals = normals * 2.0 - 1.0

    if flip_y:
        normals[..., 1] *= -1.0

    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.maximum(norm, 1e-8)

    return normals


def read_mask(
    path: Path,
    target_size: tuple[int, int],
    default_value: float,
) -> np.ndarray:
    width, height = target_size

    if not path.exists():
        return np.full((height, width), default_value, dtype=np.float32)

    mask_image = Image.open(path).convert("L")

    if mask_image.size != target_size:
        mask_image = mask_image.resize(target_size, Image.Resampling.BILINEAR)

    mask = np.asarray(mask_image).astype(np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def compute_shading(
    normals: np.ndarray,
    light_direction: np.ndarray,
    ambient: float,
    diffuse: float,
) -> np.ndarray:
    """
    Simple ambient + Lambertian directional shading.
    """
    light_direction = normalize_vector(light_direction)

    ndotl = np.sum(normals * light_direction.reshape(1, 1, 3), axis=2)
    ndotl = np.clip(ndotl, 0.0, 1.0)

    shading = ambient + diffuse * ndotl
    return np.clip(shading, 1e-4, 10.0)


def save_debug_outputs(
    debug_dir: Path,
    raw_ratio: np.ndarray,
    ratio: np.ndarray,
    final_ratio: np.ndarray,
    alpha: np.ndarray,
    subject: np.ndarray,
    background: np.ndarray,
    face: np.ndarray,
    shadow: np.ndarray,
    highlight: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Ratio maps are remapped for viewing.
    # 1.0 means no change.
    # Darker than middle means darkening.
    # Brighter than middle means brightening.
    save_gray(debug_dir / "raw_ratio_view.png", raw_ratio / 2.5)
    save_gray(debug_dir / "region_clamped_ratio_view.png", ratio / 2.5)
    save_gray(debug_dir / "final_ratio_view.png", final_ratio / 2.5)

    save_gray(debug_dir / "alpha.png", alpha)
    save_gray(debug_dir / "subject.png", subject)
    save_gray(debug_dir / "background.png", background)
    save_gray(debug_dir / "face_priority.png", face)
    save_gray(debug_dir / "shadow_protection.png", shadow)
    save_gray(debug_dir / "highlight_protection.png", highlight)


def region_aware_ratio_relight(
    image_path: Path,
    normal_path: Path,
    masks_dir: Path,
    output_path: Path,
    old_light: np.ndarray,
    new_light: np.ndarray,
    ambient: float,
    diffuse: float,
    relight_strength: float,
    subject_strength: float,
    background_strength: float,
    face_strength: float,
    shadow_protect: float,
    highlight_protect: float,
    subject_min_ratio: float,
    subject_max_ratio: float,
    background_min_ratio: float,
    background_max_ratio: float,
    flip_y: bool,
    debug_dir: Path | None,
) -> None:
    image_srgb = read_rgb(image_path)
    image_linear = srgb_to_linear(image_srgb)

    height, width = image_srgb.shape[:2]
    target_size = (width, height)

    normals = read_normal_map(
        normal_path,
        target_size=target_size,
        flip_y=flip_y,
    )

    # ------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------
    subject = read_mask(
        masks_dir / "subject.png",
        target_size,
        default_value=1.0,
    )

    # Instead of trusting background.png completely, derive it from
    # subject. This guarantees subject + background = 1.
    background = 1.0 - subject
    background = np.clip(background, 0.0, 1.0)

    face = read_mask(
        masks_dir / "face_priority.png",
        target_size,
        default_value=0.0,
    )

    shadow = read_mask(
        masks_dir / "shadow_protection.png",
        target_size,
        default_value=0.0,
    )

    highlight = read_mask(
        masks_dir / "highlight_protection.png",
        target_size,
        default_value=0.0,
    )

    # Keep face mask inside subject region.
    face = face * subject

    # ------------------------------------------------------------
    # Old and new shading
    # ------------------------------------------------------------
    old_shading = compute_shading(
        normals=normals,
        light_direction=old_light,
        ambient=ambient,
        diffuse=diffuse,
    )

    new_shading = compute_shading(
        normals=normals,
        light_direction=new_light,
        ambient=ambient,
        diffuse=diffuse,
    )

    raw_ratio = new_shading / np.maximum(old_shading, 1e-4)

    # ------------------------------------------------------------
    # Region-specific ratio limits
    # ------------------------------------------------------------
    # This is the main Stage 4 correction.
    #
    # The subject is allowed to change strongly.
    # The background is protected from becoming too dark or too bright.
    subject_ratio = np.clip(
        raw_ratio,
        subject_min_ratio,
        subject_max_ratio,
    )

    background_ratio = np.clip(
        raw_ratio,
        background_min_ratio,
        background_max_ratio,
    )

    ratio = subject_ratio * subject + background_ratio * background

    # ------------------------------------------------------------
    # Region strength
    # ------------------------------------------------------------
    # Subject gets stronger relighting.
    # Background gets weaker relighting.
    region_strength = subject_strength * subject + background_strength * background

    # Face gets gentler relighting to avoid harsh/plastic skin.
    region_strength = region_strength * (1.0 - face) + face_strength * face

    # ------------------------------------------------------------
    # Protection masks
    # ------------------------------------------------------------
    # Old shadows and highlights should resist aggressive ratio changes.
    protection = 1.0
    protection = protection * (1.0 - shadow_protect * shadow)
    protection = protection * (1.0 - highlight_protect * highlight)

    alpha = relight_strength * region_strength * protection
    alpha = np.clip(alpha, 0.0, 1.0)

    # Blend ratio toward 1.0 where alpha is weak.
    # final_ratio = 1 means original image stays unchanged.
    final_ratio = 1.0 + alpha * (ratio - 1.0)

    output_linear = image_linear * final_ratio[..., None]
    output_srgb = linear_to_srgb(output_linear)

    save_rgb(output_path, output_srgb)

    if debug_dir is not None:
        save_debug_outputs(
            debug_dir=debug_dir,
            raw_ratio=raw_ratio,
            ratio=ratio,
            final_ratio=final_ratio,
            alpha=alpha,
            subject=subject,
            background=background,
            face=face,
            shadow=shadow,
            highlight=highlight,
        )

    print(f"Region-aware relit image saved to: {output_path}")

    if debug_dir is not None:
        print(f"Debug outputs saved to: {debug_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--old-light", nargs=3, type=float, required=True)
    parser.add_argument("--new-light", nargs=3, type=float, required=True)

    parser.add_argument("--ambient", type=float, default=0.35)
    parser.add_argument("--diffuse", type=float, default=0.65)

    parser.add_argument("--relight-strength", type=float, default=1.0)
    parser.add_argument("--subject-strength", type=float, default=0.90)
    parser.add_argument("--background-strength", type=float, default=0.10)
    parser.add_argument("--face-strength", type=float, default=0.60)

    parser.add_argument("--shadow-protect", type=float, default=0.65)
    parser.add_argument("--highlight-protect", type=float, default=0.75)

    # Subject can relight strongly.
    parser.add_argument("--subject-min-ratio", type=float, default=0.45)
    parser.add_argument("--subject-max-ratio", type=float, default=2.20)

    # Background is protected.
    # This is what prevents the muddy black background issue.
    parser.add_argument("--background-min-ratio", type=float, default=0.88)
    parser.add_argument("--background-max-ratio", type=float, default=1.12)

    parser.add_argument("--flip-y", action="store_true")

    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional folder to save ratio/mask debug images.",
    )

    args = parser.parse_args()

    debug_dir = Path(args.debug_dir) if args.debug_dir is not None else None

    region_aware_ratio_relight(
        image_path=Path(args.image),
        normal_path=Path(args.normal),
        masks_dir=Path(args.masks_dir),
        output_path=Path(args.output),
        old_light=np.array(args.old_light, dtype=np.float32),
        new_light=np.array(args.new_light, dtype=np.float32),
        ambient=args.ambient,
        diffuse=args.diffuse,
        relight_strength=args.relight_strength,
        subject_strength=args.subject_strength,
        background_strength=args.background_strength,
        face_strength=args.face_strength,
        shadow_protect=args.shadow_protect,
        highlight_protect=args.highlight_protect,
        subject_min_ratio=args.subject_min_ratio,
        subject_max_ratio=args.subject_max_ratio,
        background_min_ratio=args.background_min_ratio,
        background_max_ratio=args.background_max_ratio,
        flip_y=args.flip_y,
        debug_dir=debug_dir,
    )


if __name__ == "__main__":
    main()