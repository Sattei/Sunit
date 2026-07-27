from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageSegmentation


DEFAULT_MODEL = "ZhengPeng7/BiRefNet_lite-matting"

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


def smoothstep(
    edge0: float,
    edge1: float,
    value: np.ndarray,
) -> np.ndarray:
    """
    Smoothly map values from 0 to 1 between edge0 and edge1.
    """
    if edge1 <= edge0:
        raise ValueError("edge1 must be greater than edge0")

    x = np.clip(
        (value - edge0) / (edge1 - edge0),
        0.0,
        1.0,
    )

    return x * x * (3.0 - 2.0 * x)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def choose_device(requested_device: str) -> torch.device:
    requested_device = requested_device.lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )

        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device: {requested_device}. "
        "Use auto, cpu, or cuda."
    )


def letterbox_image(
    image: Image.Image,
    size: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """
    Resize without changing aspect ratio and place the image inside
    a square canvas.

    Returns:
        canvas
        (left, top, resized_width, resized_height)
    """
    if size <= 0:
        raise ValueError("Input size must be positive")

    original_width, original_height = image.size

    scale = min(
        size / original_width,
        size / original_height,
    )

    resized_width = max(
        1,
        int(round(original_width * scale)),
    )

    resized_height = max(
        1,
        int(round(original_height * scale)),
    )

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    # Approximately the ImageNet mean colour. After normalization,
    # the padded region is close to zero rather than pure black.
    canvas = Image.new(
        "RGB",
        (size, size),
        color=(124, 116, 104),
    )

    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    canvas.paste(resized, (left, top))

    return canvas, (
        left,
        top,
        resized_width,
        resized_height,
    )


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL RGB image into an ImageNet-normalized BCHW tensor.
    """
    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)

    return tensor.contiguous()


def find_prediction_tensor(output: Any) -> torch.Tensor:
    """
    BiRefNet versions can return tensors, lists, tuples, dictionaries,
    or model-output objects. This function finds the final prediction
    tensor without tightly coupling Sunit to one return structure.
    """
    if torch.is_tensor(output):
        return output

    logits = getattr(output, "logits", None)

    if torch.is_tensor(logits):
        return logits

    if isinstance(output, dict):
        preferred_keys = (
            "logits",
            "pred",
            "preds",
            "prediction",
            "out",
        )

        for key in preferred_keys:
            value = output.get(key)

            try:
                return find_prediction_tensor(value)
            except (TypeError, ValueError):
                pass

        for value in reversed(list(output.values())):
            try:
                return find_prediction_tensor(value)
            except (TypeError, ValueError):
                pass

    if isinstance(output, (list, tuple)):
        for value in reversed(output):
            try:
                return find_prediction_tensor(value)
            except (TypeError, ValueError):
                pass

    raise TypeError(
        "Could not locate a prediction tensor in the "
        f"BiRefNet output of type {type(output)!r}"
    )


class BiRefNetMatte:
    """
    Lazy-loaded BiRefNet soft-matting adapter for Sunit.

    The returned matte is:
        shape: H x W
        dtype: float32
        range: 0.0 to 1.0
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        input_size: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.device = choose_device(device)
        self.input_size = input_size
        self.model: torch.nn.Module | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        print(f"Loading BiRefNet model: {self.model_name}")
        print(f"BiRefNet device: {self.device}")

        model = AutoModelForImageSegmentation.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model.eval()
        model.requires_grad_(False)

        if self.device.type == "cuda":
            model = model.to(
                device=self.device,
                dtype=torch.float16,
            )
        else:
            model = model.to(
                device=self.device,
                dtype=torch.float32,
            )

        self.model = model

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _predict_once(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("BiRefNet model has not been loaded")

        if self.device.type == "cuda":
            input_tensor = input_tensor.to(
                device=self.device,
                dtype=torch.float16,
            )
        else:
            input_tensor = input_tensor.to(
                device=self.device,
                dtype=torch.float32,
            )

        with torch.inference_mode():
            output = self.model(input_tensor)

        logits = find_prediction_tensor(output)

        if logits.ndim == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)

        elif logits.ndim == 3:
            logits = logits.unsqueeze(1)

        if logits.ndim != 4:
            raise RuntimeError(
                "Expected BiRefNet output with 4 dimensions, "
                f"but received shape {tuple(logits.shape)}"
            )

        # BiRefNet matting should produce one foreground channel.
        # Keep the final channel if a model variant exposes more.
        if logits.shape[1] > 1:
            logits = logits[:, -1:, :, :]

        logits = F.interpolate(
            logits.float(),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        matte = torch.sigmoid(logits)

        return matte[0, 0].detach().cpu()

    def predict(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        image = image.convert("RGB")

        original_width, original_height = image.size

        model_image, letterbox = letterbox_image(
            image,
            self.input_size,
        )

        left, top, resized_width, resized_height = letterbox

        input_tensor = image_to_tensor(model_image)

        self.load()

        try:
            matte_tensor = self._predict_once(input_tensor)

        except RuntimeError as error:
            is_cuda_oom = (
                self.device.type == "cuda"
                and "out of memory" in str(error).lower()
            )

            if not is_cuda_oom:
                raise

            print(
                "CUDA ran out of memory during BiRefNet inference. "
                "Retrying on CPU."
            )

            self.unload()
            self.device = torch.device("cpu")
            self.load()

            matte_tensor = self._predict_once(input_tensor)

        matte = matte_tensor.numpy().astype(np.float32)

        crop = matte[
            top : top + resized_height,
            left : left + resized_width,
        ]

        crop_image = Image.fromarray(
            np.clip(crop * 255.0, 0, 255).astype(np.uint8),
            mode="L",
        )

        crop_image = crop_image.resize(
            (original_width, original_height),
            Image.Resampling.LANCZOS,
        )

        alpha = np.asarray(
            crop_image,
            dtype=np.float32,
        ) / 255.0

        alpha = np.nan_to_num(
            alpha,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        return np.clip(alpha, 0.0, 1.0)


def build_relighting_masks(
    alpha: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convert the BiRefNet alpha matte into masks useful for relighting.

    alpha:
        Original soft foreground matte.

    foreground_core:
        High-confidence interior foreground.

    boundary_band:
        Soft transition region containing hair, clothing edges,
        anti-aliased pixels, and uncertain matte regions.

    relight_strength:
        Conservative relighting mask. The foreground interior receives
        full strength while boundary pixels receive reduced strength.

    background:
        Original background contribution.
    """
    alpha = np.clip(
        alpha.astype(np.float32),
        0.0,
        1.0,
    )

    foreground_core = smoothstep(
        0.55,
        0.90,
        alpha,
    )

    boundary_band = np.clip(
        alpha - foreground_core,
        0.0,
        1.0,
    )

    relight_strength = np.clip(
        foreground_core + 0.35 * boundary_band,
        0.0,
        1.0,
    )

    background = 1.0 - alpha

    return {
        "alpha": alpha,
        "foreground_core": foreground_core,
        "boundary_band": boundary_band,
        "relight_strength": relight_strength,
        "background": background,
    }


def save_grayscale(
    array: np.ndarray,
    path: Path,
) -> None:
    ensure_parent(path)

    image = Image.fromarray(
        np.clip(array * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )

    image.save(path)


def save_cutout(
    image: Image.Image,
    alpha: np.ndarray,
    path: Path,
) -> None:
    ensure_parent(path)

    rgba = image.convert("RGBA")

    alpha_image = Image.fromarray(
        np.clip(alpha * 255.0, 0, 255).astype(np.uint8),
        mode="L",
    )

    rgba.putalpha(alpha_image)
    rgba.save(path)


def save_checkerboard_preview(
    image: Image.Image,
    alpha: np.ndarray,
    path: Path,
    tile_size: int = 24,
) -> None:
    ensure_parent(path)

    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    )

    height, width = alpha.shape

    yy, xx = np.indices((height, width))

    checker_pattern = (
        (xx // tile_size + yy // tile_size) % 2
    ).astype(np.float32)

    checker_value = (
        checker_pattern * 45.0 + 190.0
    )

    checker = np.repeat(
        checker_value[:, :, None],
        3,
        axis=2,
    )

    alpha_3 = alpha[:, :, None]

    composite = (
        rgb * alpha_3
        + checker * (1.0 - alpha_3)
    )

    preview = Image.fromarray(
        np.clip(composite, 0, 255).astype(np.uint8),
        mode="RGB",
    )

    preview.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a BiRefNet soft foreground matte for "
            "the Sunit relighting pipeline."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Input RGB image.",
    )

    parser.add_argument(
        "--output-mask",
        required=True,
        type=Path,
        help="Output soft alpha mask.",
    )

    parser.add_argument(
        "--output-strength",
        type=Path,
        default=None,
        help="Optional conservative relighting-strength mask.",
    )

    parser.add_argument(
        "--output-cutout",
        type=Path,
        default=None,
        help="Optional transparent PNG cutout.",
    )

    parser.add_argument(
        "--output-preview",
        type=Path,
        default=None,
        help="Optional checkerboard preview.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face BiRefNet model name.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
        help="Square BiRefNet inference size.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(
            f"Input image does not exist: {args.image}"
        )

    image = Image.open(args.image).convert("RGB")

    estimator = BiRefNetMatte(
        model_name=args.model,
        device=args.device,
        input_size=args.input_size,
    )

    alpha = estimator.predict(image)
    masks = build_relighting_masks(alpha)

    save_grayscale(
        masks["alpha"],
        args.output_mask,
    )

    print(f"Soft alpha saved to: {args.output_mask}")

    if args.output_strength is not None:
        save_grayscale(
            masks["relight_strength"],
            args.output_strength,
        )

        print(
            "Relighting-strength mask saved to: "
            f"{args.output_strength}"
        )

    if args.output_cutout is not None:
        save_cutout(
            image,
            masks["alpha"],
            args.output_cutout,
        )

        print(f"Transparent cutout saved to: {args.output_cutout}")

    if args.output_preview is not None:
        save_checkerboard_preview(
            image,
            masks["alpha"],
            args.output_preview,
        )

        print(f"Matte preview saved to: {args.output_preview}")

    print(
        "Alpha statistics: "
        f"min={alpha.min():.4f}, "
        f"mean={alpha.mean():.4f}, "
        f"max={alpha.max():.4f}"
    )


if __name__ == "__main__":
    main()
