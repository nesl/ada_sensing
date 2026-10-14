"""
Run ParamISP exposure simulation on ImageNet-ES-Diverse samples and score
against the real captures.

For each sample image and each lighting env:
  AE shot --inverse ISP--> pseudo-RAW --scale--> forward ISP @ each of the
  27 param_control configs; compare vs the real capture of that config.

The AE shots carry no EXIF, so the AE exposure product E = t*ISO/N^2 is
estimated per environment by median luminance-matching against the real
param_13 captures (ISO 250, 1/60s, f/9.0) over a separate calibration set.
That is one scalar per environment; no other ground truth enters the sim.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paramisp_sim as P  # noqa: E402

DATA = Path("/Users/malikh/Desktop/ada_sensing/data/ImageNet-ES-Diverse/es-diverse-test")
OUT = Path(__file__).resolve().parent / "results"
ENVS = ["l1", "l2", "l3", "l4", "l6", "l7"]
N_SAMPLES = 10
N_CALIB = 5
CALIB_PARAM = 13  # ISO 250, 1/60, f9.0 -- middle of the grid
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ISO_SEQ = [250, 2000, 16000]
SS_SEQ = [1 / 4, 1 / 60, 1 / 1000]
AP_SEQ = [5.0, 9.0, 16.0]


def param_config(k: int) -> tuple[float, float, float]:
    """param_k (1-indexed) -> (shutter_s, iso, f_number).

    Order from Edw2n/ImageNet-ES settings/grid-options-3x3x3.csv:
    ISO fastest, then shutter, then aperture."""
    i = k - 1
    return SS_SEQ[(i // 3) % 3], ISO_SEQ[i % 3], AP_SEQ[i // 9]


def exposure_product(t, iso, n):
    return t * iso / n ** 2


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def mean_linear(path: Path) -> float:
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return float(srgb_to_linear(a).mean())


def list_samples():
    """(synset, stem) pairs present in AE + all 27 params across all envs."""
    root = DATA / "param_control" / "l1" / "param_1"
    candidates = []
    for syn_dir in sorted(root.iterdir()):
        if not syn_dir.is_dir():
            continue
        for img in sorted(syn_dir.glob("*.JPEG")):
            candidates.append((syn_dir.name, img.stem))
    ok = []
    for syn, stem in candidates:
        present = all(
            (DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG").exists()
            for env in ENVS for k in (1, 14, 27)
        ) and all(
            (DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG").exists()
            for env in ENVS
        )
        if ok and ok[-1][0] == syn:  # at most one image per synset, for diversity
            continue
        if present:
            ok.append((syn, stem))
        if len(ok) >= N_SAMPLES + N_CALIB:
            break
    assert len(ok) >= N_SAMPLES + N_CALIB, f"only {len(ok)} complete samples found"
    return ok[:N_CALIB], ok[N_CALIB:N_CALIB + N_SAMPLES]


def _linear(path: Path) -> np.ndarray:
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return srgb_to_linear(a)


# candidate reference configs spanning the exposure range; per env we use only
# those whose real capture is well-exposed (else the JPEG quantization floor or
# highlight clipping corrupts the luminance ratio)
REF_PARAMS = [2, 4, 5, 13, 14, 15, 23, 24]


def calibrate_ae_exposure(calib):
    """Estimate E_AE per env from per-pixel linear ratios vs well-exposed refs.

    For pixels far from the black floor and from clipping, lin_AE/lin_ref ~= E_AE/E_ref.
    Median over pixels, refs and calibration images."""
    out = {}
    for env in ENVS:
        ratios = []
        for syn, stem in calib:
            ae = _linear(DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG")
            for k in REF_PARAMS:
                p = DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG"
                if not p.exists():
                    continue
                ref = _linear(p)
                if not (0.05 < ref.mean() < 0.55):  # skip under/over-exposed refs
                    continue
                if ref.shape != ae.shape:
                    continue
                mask = (ref > 0.02) & (ref < 0.8) & (ae > 0.02) & (ae < 0.8)
                if mask.sum() < 500:
                    continue
                t, iso, n = param_config(k)
                r = np.median(ae[mask] / ref[mask])
                ratios.append(exposure_product(t, iso, n) * float(r))
        assert ratios, f"no valid calibration refs for {env}"
        out[env] = float(np.median(ratios))
    return out


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


@torch.no_grad()
def refine_ae_exposure(inv, fwd, calib, e_init):
    """Refine E_AE per env by matching simulated vs real mean brightness through
    the simulator (folds the unknown camera tone curve into the scalar). Uses
    calibration images only; secant iterations in log space."""
    out = {}
    for env in ENVS:
        pairs = []  # (ae_tensor, ref_k, real_mean)
        for syn, stem in calib[:3]:
            ae = P.load_image(str(DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG"), 384)
            for k in REF_PARAMS:
                p = DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG"
                if not p.exists():
                    continue
                real = P.load_image(str(p), 384)
                if 0.05 < real.mean().item() < 0.55:
                    pairs.append((ae, k, real.mean().item()))
            if len(pairs) >= 4:
                break

        def err(e_ae: float) -> float:
            tot = 0.0
            for ae, k, real_mean in pairs:
                n_ae, iso_ae = 5.0, 250.0
                t_ae = float(np.clip(e_ae * n_ae ** 2 / iso_ae, 1e-5, 30.0))
                raw = inv(P.make_batch(ae.to(DEVICE), None, t_ae, iso_ae, n_ae, device=DEVICE))
                t, iso, n = param_config(k)
                ratio = exposure_product(t, iso, n) / e_ae
                sim = fwd(P.make_batch(ae.to(DEVICE), (raw * ratio).clip(0, 1), t, iso, n, device=DEVICE))
                tot += np.log(max(sim.mean().item(), 1e-4)) - np.log(max(real_mean, 1e-4))
            return tot / len(pairs)

        # secant in log(E) space
        x0, x1 = np.log(e_init[env]), np.log(e_init[env]) - 0.7
        f0, f1 = err(np.exp(x0)), err(np.exp(x1))
        for _ in range(4):
            if abs(f1 - f0) < 1e-4:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x2 = np.clip(x2, x1 - 1.5, x1 + 1.5)
            x0, f0, x1 = x1, f1, x2
            f1 = err(np.exp(x1))
        out[env] = float(np.exp(x1))
        print(f"  {env}: E_AE {e_init[env]:.3f} -> {out[env]:.3f} (log-err {f1:+.3f})", flush=True)
    return out


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    calib, samples = list_samples()
    print(f"calib: {[s[1][-9:] for s in calib]}")
    print(f"samples: {[s[1][-9:] for s in samples]}")

    e_ae = calibrate_ae_exposure(calib)
    print("E_AE initial:", {k: round(v, 4) for k, v in e_ae.items()})

    W = "/Users/malikh/Desktop/ada_sensing/ParamISP/weights"
    inv = P.load_model(f"{W}/pre_training/inverse.ckpt", inverse=True, device=DEVICE)
    fwd = P.load_model(f"{W}/pre_training/forward.ckpt", inverse=False, device=DEVICE)

    e_ae = refine_ae_exposure(inv, fwd, calib, e_ae)

    metrics = []
    for si, (syn, stem) in enumerate(samples):
        for env in ENVS:
            ae_path = DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG"
            ae = P.load_image(str(ae_path), max_side=384)

            # nominal AE params: assume widest aperture, base ISO; recover shutter from E_AE
            n_ae, iso_ae = 5.0, 250.0
            t_ae = float(np.clip(e_ae[env] * n_ae ** 2 / iso_ae, 1e-5, 30.0))
            ae_params = (t_ae, iso_ae, n_ae)

            # invert once per (sample, env)
            batch = P.make_batch(ae.to(DEVICE), None, t_ae, iso_ae, n_ae, device=DEVICE)
            raw_ae = inv(batch)

            sample_dir = OUT / f"{si:02d}_{syn}_{stem}" / env
            sample_dir.mkdir(parents=True, exist_ok=True)
            P.to_pil(ae).save(sample_dir / "AE.jpg", quality=92)

            for k in range(1, 28):
                t, iso, n = param_config(k)
                ratio = exposure_product(t, iso, n) / e_ae[env]
                raw_t = (raw_ae * ratio).clip(0, 1)
                batch = P.make_batch(ae.to(DEVICE), raw_t, t, iso, n, device=DEVICE)
                sim = fwd(batch).clip(0, 1).cpu()

                real_path = DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG"
                real = P.load_image(str(real_path), max_side=384)
                if real.shape != sim.shape:  # guard: identical rig, should match
                    real = torch.nn.functional.interpolate(real, size=sim.shape[-2:],
                                                           mode="bilinear", align_corners=False)

                P.to_pil(sim).save(sample_dir / f"sim_param_{k}.jpg", quality=92)
                P.to_pil(real).save(sample_dir / f"real_param_{k}.jpg", quality=92)

                metrics.append({
                    "sample": f"{syn}/{stem}", "env": env, "param": k,
                    "ss": t, "iso": iso, "f": n,
                    "psnr_sim": round(psnr(sim, real), 3),
                    "psnr_ae_baseline": round(psnr(ae, real), 3),
                    "l1_sim": round((sim - real).abs().mean().item(), 5),
                    "mean_sim": round(sim.mean().item(), 4),
                    "mean_real": round(real.mean().item(), 4),
                })
        done = len(metrics)
        print(f"[{si+1}/{len(samples)}] {syn}/{stem}: {done} comparisons", flush=True)

    with open(OUT / "metrics.json", "w") as f:
        json.dump({"e_ae": e_ae, "device": DEVICE, "metrics": metrics}, f, indent=1)

    arr_sim = np.array([m["psnr_sim"] for m in metrics])
    arr_ae = np.array([m["psnr_ae_baseline"] for m in metrics])
    print(f"\noverall: sim PSNR {arr_sim.mean():.2f} dB | AE-copy baseline {arr_ae.mean():.2f} dB")
    for env in ENVS:
        s = np.array([m["psnr_sim"] for m in metrics if m["env"] == env])
        b = np.array([m["psnr_ae_baseline"] for m in metrics if m["env"] == env])
        print(f"  {env}: sim {s.mean():5.2f} dB vs baseline {b.mean():5.2f} dB")


if __name__ == "__main__":
    main()
