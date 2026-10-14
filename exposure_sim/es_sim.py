"""
Physically-grounded exposure simulator for ImageNet-ES-Diverse.

Takes an auto-exposure (AE) capture and predicts what the same scene looks like
under each of the 27 param_control settings. Validated against the real captures,
which is the point: once the error is characterised here, the same simulator can be
applied to video data where no ground-truth config sweep exists.

Config space (from policy_network/.../plot_oracle_soft_label_param_heatmaps.py):
    aperture in {f/5.0, f/9.0, f/16}, shutter in {1/4, 1/60, 1/1000}, ISO in {250, 2000, 16000}
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

APERTURES = [5.0, 9.0, 16.0]
SS_VALUES = [1 / 4, 1 / 60, 1 / 1000]
ISO_VALUES = [250, 2000, 16000]


@dataclass(frozen=True)
class Config:
    aperture: float   # f-number
    shutter: float    # seconds
    iso: int

    @property
    def log2_exposure(self) -> float:
        """Relative exposure in stops. Light on sensor ~ t / N^2, then analog gain ~ ISO."""
        return np.log2(self.shutter * self.iso / (self.aperture ** 2))


def config_table() -> list[Config]:
    """The 27 param_control settings. Order must match option_id in the manifests."""
    return [Config(a, s, i) for a, s, i in itertools.product(APERTURES, ISO_VALUES, SS_VALUES)]


# ---------------------------------------------------------------- sRGB <-> linear

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


# ---------------------------------------------------------------- noise model
# Sensor model in normalised linear units. With e electrons and analog gain g ~ ISO,
# the digital value is D = g*e/FWC, so:
#   shot noise   var_D = D * k_shot * ISO      (Poisson, scaled through the gain)
#   read noise   var_D = (k_read * ISO)^2      (analog read noise, amplified by gain)
# k_shot and k_read are sensor constants, fitted from the dataset by fit_noise().

@dataclass
class NoiseParams:
    k_shot: float = 1.0e-8
    k_read: float = 2.0e-7

    def sigma(self, signal: np.ndarray, iso: int) -> np.ndarray:
        var = np.clip(signal, 0.0, None) * self.k_shot * iso + (self.k_read * iso) ** 2
        return np.sqrt(var)


def simulate(
    ae_img: np.ndarray,
    ae_cfg: Config,
    target_cfg: Config,
    noise: NoiseParams | None = None,
    rng: np.random.Generator | None = None,
    add_noise: bool = True,
) -> np.ndarray:
    """
    ae_img: HxWx3 uint8 or float in [0,1], sRGB.
    Returns the simulated capture at target_cfg, same shape/dtype convention (float [0,1]).

    Highlight clipping is applied in linear space before re-encoding -- this is what
    makes the overexposed configs (1/4s at ISO 16000) look correct rather than merely bright.
    """
    if ae_img.dtype == np.uint8:
        ae_img = ae_img.astype(np.float32) / 255.0
    rng = rng or np.random.default_rng(0)
    noise = noise or NoiseParams()

    lin = srgb_to_linear(ae_img)

    # Undo the AE capture's own analog gain to recover a gain-free radiance estimate,
    # then apply the target's exposure. Splitting it this way matters because the noise
    # depends on the target ISO, not on the ratio.
    ratio = 2.0 ** (target_cfg.log2_exposure - ae_cfg.log2_exposure)
    out = lin * ratio

    if add_noise:
        out = out + rng.normal(0.0, 1.0, out.shape) * noise.sigma(out, target_cfg.iso)

    out = np.clip(out, 0.0, 1.0)
    return linear_to_srgb(out).astype(np.float32)


def fit_noise(pairs, iso_grid=ISO_VALUES) -> NoiseParams:
    """
    Fit k_shot/k_read from real captures.

    pairs: iterable of (image_float01_srgb, Config) for flat/low-detail regions.
    Estimates local variance vs local mean in linear space per ISO and does a
    least-squares fit of var = k_shot*ISO*mean + (k_read*ISO)^2.
    """
    import numpy.linalg as la

    rows, targets = [], []
    for img, cfg in pairs:
        lin = srgb_to_linear(img)
        # 8x8 tiles; use tiles with low gradient so variance is noise, not texture
        h, w, _ = lin.shape
        th, tw = h // 8 * 8, w // 8 * 8
        tiles = lin[:th, :tw].reshape(th // 8, 8, tw // 8, 8, 3).transpose(0, 2, 1, 3, 4)
        tiles = tiles.reshape(-1, 8 * 8, 3)
        mean = tiles.mean(axis=1)
        var = tiles.var(axis=1)
        # keep unsaturated, low-texture tiles
        keep = (mean < 0.9).all(axis=1) & (var < np.percentile(var, 40))
        for m, v in zip(mean[keep].ravel(), var[keep].ravel()):
            rows.append([cfg.iso * m, cfg.iso ** 2])
            targets.append(v)

    if not rows:
        return NoiseParams()
    A = np.asarray(rows)
    b = np.asarray(targets)
    sol, *_ = la.lstsq(A, b, rcond=None)
    k_shot = max(sol[0], 1e-12)
    k_read = np.sqrt(max(sol[1], 1e-16))
    return NoiseParams(k_shot=float(k_shot), k_read=float(k_read))
