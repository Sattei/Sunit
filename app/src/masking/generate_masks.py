import argparse
from pathlib import Path

import numpy as np

from mask_utils import (
    ellipse_mask,
    gaussian_blur,
    luminance,
    read_rgb,
    save_gray,
    smoothstep,
    srgb_to_linear,
)


def generate_masks(
    image_path: Path,
    output_dir: Path,
    subject_cx: float,
    subject_cy: float,
    subject_rx: float,
    subject_ry: float,
) -> None:
    image_srgb = read_rgb(image_path)
    image_linear = srgb_to_linear(image_srgb)

    height, width = image_srgb.shape[:2]

    lum = luminance(image_linear)

    # ------------------------------------------------------------
    # 1. Subject mask
    # ------------------------------------------------------------
    # For now, this is a portrait-friendly soft center mask.
    # Later, we can replace this with BiRefNet / SAM 2 / rembg.
    subject = ellipse_mask(
        height=height,
        width=width,
        cx=subject_cx,
        cy=subject_cy,
        rx=subject_rx,
        ry=subject_ry,
        softness=0.22,
    )

    subject = gaussian_blur(subject, radius=max(3, int(min(height, width) * 0.01)))
    subject = np.clip(subject, 0.0, 1.0)

    background = 1.0 - subject

    # ------------------------------------------------------------
    # 2. Face priority mask
    # ------------------------------------------------------------
    # Not actual face detection yet.
    # This is only a soft upper-center portrait prior.
    # It helps keep facial relighting gentler and more natural.
    face_priority = ellipse_mask(
        height=height,
        width=width,
        cx=0.50,
        cy=0.34,
        rx=0.22,
        ry=0.20,
        softness=0.35,
    )

    face_priority = face_priority * subject
    face_priority = gaussian_blur(
        face_priority,
        radius=max(5, int(min(height, width) * 0.018)),
    )
    face_priority = np.clip(face_priority, 0.0, 1.0)

    # ------------------------------------------------------------
    # 3. Shadow protection mask
    # ------------------------------------------------------------
    # We detect areas that are darker than their local neighborhood.
    # These are likely old cast shadows / dark folds / baked lighting.
    local_blur_radius = max(9, int(min(height, width) * 0.035))
    local_lum = gaussian_blur(lum, radius=local_blur_radius)

    local_ratio = lum / np.maximum(local_lum, 1e-6)

    local_shadow = 1.0 - smoothstep(0.68, 0.98, local_ratio)
    absolute_shadow = 1.0 - smoothstep(0.10, 0.36, lum)

    shadow_protection = np.maximum(local_shadow * 0.85, absolute_shadow * 0.55)
    shadow_protection = gaussian_blur(
        shadow_protection,
        radius=max(3, int(min(height, width) * 0.008)),
    )
    shadow_protection = np.clip(shadow_protection, 0.0, 1.0)

    # ------------------------------------------------------------
    # 4. Highlight protection mask
    # ------------------------------------------------------------
    # Bright regions should not be amplified too much.
    highlight_protection = smoothstep(0.62, 0.92, lum)
    highlight_protection = gaussian_blur(
        highlight_protection,
        radius=max(3, int(min(height, width) * 0.008)),
    )
    highlight_protection = np.clip(highlight_protection, 0.0, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_gray(output_dir / "subject.png", subject)
    save_gray(output_dir / "background.png", background)
    save_gray(output_dir / "face_priority.png", face_priority)
    save_gray(output_dir / "shadow_protection.png", shadow_protection)
    save_gray(output_dir / "highlight_protection.png", highlight_protection)

    print(f"Masks saved to: {output_dir}")
    print("Generated:")
    print("  subject.png")
    print("  background.png")
    print("  face_priority.png")
    print("  shadow_protection.png")
    print("  highlight_protection.png")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output-dir", required=True, help="Mask output folder")

    parser.add_argument("--subject-cx", type=float, default=0.50)
    parser.add_argument("--subject-cy", type=float, default=0.52)
    parser.add_argument("--subject-rx", type=float, default=0.38)
    parser.add_argument("--subject-ry", type=float, default=0.50)

    args = parser.parse_args()

    generate_masks(
        image_path=Path(args.image),
        output_dir=Path(args.output_dir),
        subject_cx=args.subject_cx,
        subject_cy=args.subject_cy,
        subject_rx=args.subject_rx,
        subject_ry=args.subject_ry,
    )


if __name__ == "__main__":
    main()