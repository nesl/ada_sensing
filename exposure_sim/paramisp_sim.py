"""
Exposure-config simulation on ImageNet-ES-Diverse via ParamISP (CVPR 2024).

Pipeline per image:
  AE sRGB --inverse ParamISP(AE params)--> pseudo-RAW
          --scale by exposure ratio (t*ISO/N^2)--> target RAW
          --forward ParamISP(target params)--> simulated sRGB @ target config

Uses the official pre-training (multi-camera) checkpoints; the ES-Diverse camera
is unseen, so results measure cross-camera generalization -- which is the point:
validate here against real captures before trusting the simulator on video data.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent / "ParamISP" / "ParamISP"
sys.path.insert(0, str(REPO))

# Shim: the repo targets pytorch-lightning 1.6; we run 1.9.5. Alias names that
# moved since then. Only their (unused) ImageLogger needs these, never our path.
import pytorch_lightning.utilities.logger as _pl_ulog  # noqa: E402
if not hasattr(_pl_ulog, "_add_prefix"):
    from lightning_fabric.utilities.logger import _add_prefix, _convert_params
    _pl_ulog._add_prefix = _add_prefix
    _pl_ulog._convert_params = _convert_params
import pytorch_lightning.loggers.base as _pl_lbase  # noqa: E402
import pytorch_lightning.loggers.logger as _pl_llog  # noqa: E402
_pl_lbase.LightningLoggerBase = _pl_llog.Logger
_pl_lbase.rank_zero_experiment = _pl_llog.rank_zero_experiment

import models.paramisp as mp  # noqa: E402
import layers.bayer  # noqa: E402
from models.utils.arg import Namespace  # noqa: E402

# ImageNet-ES-Diverse param_control grid (from ada_sensing policy_network scripts)
APERTURES = [5.0, 9.0, 16.0]
SS_VALUES = [1 / 4, 1 / 60, 1 / 1000]
ISO_VALUES = [250, 2000, 16000]

# Metadata the JPEGs cannot provide. Chosen by round-trip self-consistency:
# identity round-trip (same params in/out) reaches ~27.5dB PSNR with unit WB and a
# moderate row-normalized CCM, the best of a WB x CCM-strength sweep on natural images.
WHITE_BALANCE = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_s = 0.3
COLOR_MATRIX = (np.eye(3, dtype=np.float32) * (1 + 2 * _s) - _s)
COLOR_MATRIX /= COLOR_MATRIX.sum(axis=1, keepdims=True)
BAYER = torch.tensor([[[0, 1], [1, 2]]])  # RGGB, encoded as channel indices


def load_model(ckpt_path: str, inverse: bool, device: str = "cpu") -> mp.ParamISP:
    args = Namespace(inverse=inverse)
    model = mp.ParamISP(args)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    if unexpected:
        print(f"note: {len(unexpected)} unexpected keys ignored (e.g. {unexpected[:2]})")
    return model.to(device).eval()


def make_batch(rgb: torch.Tensor | None, raw: torch.Tensor | None,
               exposure_time: float, iso: float, f_number: float,
               focal_length: float = 4.4, device: str = "cpu") -> dict:
    """Build an ImageDataBatch-compatible dict for a single image."""
    n = 1
    if rgb is None:
        rgb = torch.zeros(n, 3, 2, 2)
    if raw is None:
        raw = torch.zeros(n, 1, 2, 2)
    return {
        "rgb": rgb.to(device),
        "raw": raw.to(device),
        "bayer_pattern": BAYER.expand(n, -1, -1).unsqueeze(1).to(device),
        "white_balance": torch.from_numpy(WHITE_BALANCE).unsqueeze(0).to(device),
        "color_matrix": torch.from_numpy(COLOR_MATRIX).unsqueeze(0).to(device),
        "focal_length": torch.tensor([focal_length], device=device),
        "f_number": torch.tensor([f_number], device=device),
        "exposure_time": torch.tensor([exposure_time], device=device),
        "iso_sensitivity": torch.tensor([iso], device=device),
        "camera_name": ["UNKNOWN/UNKNOWN"],
        "inst_id": ["es-diverse"],
        "index": torch.tensor([0], device=device),
        "full_image_size": [(rgb.shape[-2], rgb.shape[-1])],
        "window_top_left": [(0, 0)],
        "transform": [(False, False, 0)],
        "quantized_level": torch.tensor([[1023.0, 1023.0, 1023.0]], device=device),
    }


def exposure_ratio(t1, iso1, n1, t2, iso2, n2) -> float:
    """Light ratio between two configs: exposure ~ t * ISO / N^2."""
    return (t2 * iso2 / n2 ** 2) / (t1 * iso1 / n1 ** 2)


@torch.no_grad()
def simulate(inv_model, fwd_model, ae_rgb: torch.Tensor,
             ae_params: tuple[float, float, float],
             target_params: tuple[float, float, float],
             device: str = "cpu") -> torch.Tensor:
    """ae_rgb: (1,3,H,W) in [0,1]. params: (exposure_time_s, iso, f_number)."""
    t1, iso1, n1 = ae_params
    t2, iso2, n2 = target_params

    batch = make_batch(ae_rgb, None, t1, iso1, n1, device=device)
    raw = inv_model(batch)  # (1,1,H,W) mosaic

    raw = (raw * exposure_ratio(t1, iso1, n1, t2, iso2, n2)).clip(0, 1)

    batch = make_batch(ae_rgb, raw, t2, iso2, n2, device=device)
    out = fwd_model(batch).clip(0, 1)
    return out


def load_image(path: str, max_side: int = 512) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    # multiples of 8: bayer mosaic needs even dims, the U-Nets need /4 twice
    img = img.resize((max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)), Image.LANCZOS)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0)


def to_pil(x: torch.Tensor) -> Image.Image:
    a = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)
