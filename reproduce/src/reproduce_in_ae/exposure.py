from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from PIL import Image


FIXED_EV_VALUES: tuple[float, ...] = tuple(
    -4.0 + 0.5 * index for index in range(17)
)
TARGET_LUMINANCE_VALUES: tuple[float, ...] = (
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.45,
    0.65,
    0.85,
    0.95,
)
EXPOSURE_MODES: tuple[str, ...] = ("fixed_ev", "target_mean_luminance")
REC709_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
_SRGB_U8_LUT = np.where(
    np.arange(256, dtype=np.float32) / 255.0 <= 0.04045,
    (np.arange(256, dtype=np.float32) / 255.0) / 12.92,
    (
        (np.arange(256, dtype=np.float32) / 255.0 + 0.055)
        / 1.055
    )
    ** 2.4,
).astype(np.float32)


@dataclass(frozen=True)
class ExposureSpec:
    mode: str
    value: float

    def __post_init__(self) -> None:
        if self.mode not in EXPOSURE_MODES:
            raise ValueError(
                f"Unknown exposure mode {self.mode!r}; expected one of {EXPOSURE_MODES}"
            )
        if not math.isfinite(self.value):
            raise ValueError("Exposure value must be finite")
        if self.mode == "target_mean_luminance" and not 0.0 < self.value <= 1.0:
            raise ValueError("Target mean luminance must be in (0, 1]")

    @property
    def tag(self) -> str:
        prefix = "ev" if self.mode == "fixed_ev" else "target_y"
        return f"{prefix}_{format_value(self.value)}"


@dataclass(frozen=True)
class ExposureMetadata:
    gain: float
    effective_ev: float
    original_mean_luminance: float
    achieved_mean_luminance: float
    near_black_fraction: float
    near_white_fraction: float
    any_channel_zero_fraction: float
    any_channel_saturated_fraction: float


def format_value(value: float) -> str:
    if abs(value) < 5e-12:
        value = 0.0
    text = f"{value:+.6f}".rstrip("0").rstrip(".")
    return text.replace("+", "p").replace("-", "m").replace(".", "p")


def default_specs() -> tuple[ExposureSpec, ...]:
    return tuple(
        [ExposureSpec("fixed_ev", value) for value in FIXED_EV_VALUES]
        + [
            ExposureSpec("target_mean_luminance", value)
            for value in TARGET_LUMINANCE_VALUES
        ]
    )


def srgb_u8_to_linear(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise TypeError(f"Expected uint8 RGB input, found {array.dtype}")
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] RGB input, found shape {array.shape}")
    return _SRGB_U8_LUT[array]


def linear_rgb_to_srgb_u8(linear_rgb: np.ndarray) -> np.ndarray:
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def linear_luminance(linear_rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(linear_rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] RGB input, found shape {array.shape}")
    return np.sum(array * REC709_WEIGHTS, axis=-1, dtype=np.float32)


def mean_linear_luminance(rgb_u8: np.ndarray) -> float:
    return float(linear_luminance(srgb_u8_to_linear(rgb_u8)).mean(dtype=np.float64))


def exposure_gain(spec: ExposureSpec, current_mean_luminance: float) -> float:
    if spec.mode == "fixed_ev":
        return float(2.0**spec.value)
    return float(spec.value / max(float(current_mean_luminance), 1e-6))


def apply_exposure(
    image: Image.Image,
    spec: ExposureSpec,
    current_mean_luminance: float | None = None,
    original_metrics: Mapping[str, float] | None = None,
) -> tuple[Image.Image, ExposureMetadata]:
    rgb_image = image.convert("RGB")
    if current_mean_luminance is None:
        current_mean_luminance = mean_linear_luminance(np.asarray(rgb_image))
    gain = exposure_gain(spec, current_mean_luminance)
    effective_ev = float(math.log2(gain))

    if spec.mode == "fixed_ev" and spec.value == 0.0:
        metrics = original_metrics or {}
        return rgb_image, ExposureMetadata(
            gain=1.0,
            effective_ev=0.0,
            original_mean_luminance=float(current_mean_luminance),
            achieved_mean_luminance=float(current_mean_luminance),
            near_black_fraction=float(metrics.get("near_black_fraction", float("nan"))),
            near_white_fraction=float(metrics.get("near_white_fraction", float("nan"))),
            any_channel_zero_fraction=float(
                metrics.get("any_channel_zero_fraction", float("nan"))
            ),
            any_channel_saturated_fraction=float(
                metrics.get("any_channel_saturated_fraction", float("nan"))
            ),
        )

    linear = srgb_u8_to_linear(np.asarray(rgb_image))
    adjusted = np.clip(linear * gain, 0.0, 1.0)
    luminance = linear_luminance(adjusted)
    metadata = ExposureMetadata(
        gain=gain,
        effective_ev=effective_ev,
        original_mean_luminance=float(current_mean_luminance),
        achieved_mean_luminance=float(luminance.mean(dtype=np.float64)),
        near_black_fraction=float(np.mean(luminance <= 0.001)),
        near_white_fraction=float(np.mean(luminance >= 0.99)),
        any_channel_zero_fraction=float(np.mean(np.any(adjusted <= 0.0, axis=-1))),
        any_channel_saturated_fraction=float(
            np.mean(np.any(adjusted >= 1.0, axis=-1))
        ),
    )
    adjusted_image = Image.fromarray(linear_rgb_to_srgb_u8(adjusted), mode="RGB")
    return adjusted_image, metadata


def load_luminance_index(path: Path) -> Dict[str, Dict[str, float]]:
    index: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            resolved = str(Path(row["path"]).resolve())
            if resolved in index:
                raise ValueError(f"Duplicate luminance record for {resolved}")
            index[resolved] = {
                "mean_luminance": float(row["mean_luminance"]),
                "near_black_fraction": float(row["near_black_fraction"]),
                "near_white_fraction": float(row["near_white_fraction"]),
                "any_channel_zero_fraction": float(
                    row["any_channel_zero_fraction"]
                ),
                "any_channel_saturated_fraction": float(
                    row["any_channel_saturated_fraction"]
                ),
            }
    if not index:
        raise ValueError(f"No luminance records found in {path}")
    return index
