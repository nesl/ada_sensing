"""
Post-hoc calibration of ParamISP sims to the ES-Diverse camera.

Fits, per lighting environment, a global correction sim -> real:
  1) a 3x3 color matrix in linear RGB          (WB/CCM mismatch)
  2) a per-channel monotonic tone LUT in sRGB  (tone-curve mismatch)

Fit ONLY on the 5 calibration images (all 27 configs each); applied to the
10 evaluation samples and re-scored. No eval ground truth enters the fit.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import paramisp_sim as P
import run_es_diverse as R

HERE = Path(__file__).resolve().parent
CALIB_OUT = HERE / "results_calib"
RES = HERE / "results"
DEVICE = R.DEVICE

E_AE = json.load(open(RES / "metrics.json"))["e_ae"]


def lin(a):  # sRGB [0,1] float array -> linear
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def imread(p):
    return np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0


@torch.no_grad()
def simulate_calib_set(calib):
    """Run the identical pipeline on calibration images; save sim JPEGs."""
    W = "/Users/malikh/Desktop/ada_sensing/ParamISP/weights"
    inv = P.load_model(f"{W}/pre_training/inverse.ckpt", inverse=True, device=DEVICE)
    fwd = P.load_model(f"{W}/pre_training/forward.ckpt", inverse=False, device=DEVICE)
    for syn, stem in calib:
        for env in R.ENVS:
            d = CALIB_OUT / f"{syn}_{stem}" / env
            if (d / "sim_param_27.jpg").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            ae = P.load_image(str(R.DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG"), 384)
            n_ae, iso_ae = 5.0, 250.0
            t_ae = float(np.clip(E_AE[env] * n_ae ** 2 / iso_ae, 1e-5, 30.0))
            raw = inv(P.make_batch(ae.to(DEVICE), None, t_ae, iso_ae, n_ae, device=DEVICE))
            for k in range(1, 28):
                t, iso, n = R.param_config(k)
                ratio = R.exposure_product(t, iso, n) / E_AE[env]
                sim = fwd(P.make_batch(ae.to(DEVICE), (raw * ratio).clip(0, 1), t, iso, n, device=DEVICE)).clip(0, 1)
                P.to_pil(sim.cpu()).save(d / f"sim_param_{k}.jpg", quality=95)
        print(f"simulated calib {syn}/{stem}", flush=True)


def fit_env(env, calib, n_pix=200_000, n_knots=17):
    """Fit 3x3 linear matrix + per-channel monotonic sRGB LUT for one env."""
    S, T = [], []
    for syn, stem in calib:
        d = CALIB_OUT / f"{syn}_{stem}" / env
        for k in range(1, 28):
            sp = d / f"sim_param_{k}.jpg"
            rp = R.DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG"
            if not sp.exists() or not rp.exists():
                continue
            s = imread(sp)
            r = imread(rp)
            if r.shape != s.shape:
                im = Image.open(rp).convert("RGB").resize((s.shape[1], s.shape[0]), Image.LANCZOS)
                r = np.asarray(im, np.float32) / 255.0
            S.append(s.reshape(-1, 3))
            T.append(r.reshape(-1, 3))
    S = np.concatenate(S)
    T = np.concatenate(T)
    idx = np.random.default_rng(0).choice(len(S), min(n_pix, len(S)), replace=False)
    S, T = S[idx], T[idx]

    # 1) 3x3 matrix in linear space
    Sl, Tl = lin(S), lin(T)
    Mt, *_ = np.linalg.lstsq(Sl, Tl, rcond=None)  # (3,3): Sl @ Mt ~= Tl
    inter = srgb(np.clip(Sl @ Mt, 0, 1))

    # 2) per-channel monotonic LUT (piecewise linear, isotonic via cummax)
    knots = np.linspace(0, 1, n_knots)
    luts = np.zeros((3, n_knots), np.float32)
    for c in range(3):
        xc, yc = inter[:, c], T[:, c]
        vals = []
        for i, kn in enumerate(knots):
            m = np.abs(xc - kn) < (0.75 / (n_knots - 1))
            vals.append(yc[m].mean() if m.sum() > 50 else np.nan)
        v = np.array(vals)
        # fill gaps by interpolation, then enforce monotonicity
        ok = ~np.isnan(v)
        v = np.interp(knots, knots[ok], v[ok])
        # pin the black point: per-config noise floors differ, and mapping true
        # black to the average floor wrecks the near-black cells' PSNR
        v[0] = 0.0
        luts[c] = np.maximum.accumulate(v)
    return Mt.astype(np.float32), knots.astype(np.float32), luts


def apply_cal(img, Mt, knots, luts):
    x = srgb(np.clip(lin(img) @ Mt, 0, 1))
    out = np.stack([np.interp(x[..., c], knots, luts[c]) for c in range(3)], -1)
    return np.clip(out, 0, 1).astype(np.float32)


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 10 * np.log10(1 / max(m, 1e-10))


def main():
    calib, _samples = R.list_samples()
    simulate_calib_set(calib)

    cal = {env: fit_env(env, calib) for env in R.ENVS}
    print("fitted per-env corrections")

    rows = []
    for sd in sorted(RES.iterdir()):
        if not sd.is_dir():
            continue
        for env in R.ENVS:
            d = sd / env
            Mt, knots, luts = cal[env]
            for k in range(1, 28):
                s = imread(d / f"sim_param_{k}.jpg")
                r = imread(d / f"real_param_{k}.jpg")
                sc = apply_cal(s, Mt, knots, luts)
                Image.fromarray((sc * 255).round().astype(np.uint8)).save(
                    d / f"simcal_param_{k}.jpg", quality=92)
                rows.append({"sample": sd.name, "env": env, "param": k,
                             "psnr_raw": psnr(s, r), "psnr_cal": psnr(sc, r),
                             "real_mean": float(r.mean())})
        print(f"applied to {sd.name}", flush=True)

    a = np.array([[x["psnr_raw"], x["psnr_cal"], x["real_mean"]] for x in rows])
    print(f"\noverall: raw {a[:,0].mean():.2f} dB -> calibrated {a[:,1].mean():.2f} dB")
    for lo, hi, tag in [(0, .08, "near-black"), (.08, .3, "dark-mid"), (.3, .6, "well-exposed"), (.6, 1, "bright/blown")]:
        m = (a[:, 2] >= lo) & (a[:, 2] < hi)
        if m.sum():
            print(f"  {tag:13s} n={int(m.sum()):4d}  raw {a[m,0].mean():5.2f} -> cal {a[m,1].mean():5.2f}  ({a[m,1].mean()-a[m,0].mean():+.2f})")

    json.dump({"rows": rows}, open(RES / "metrics_posthoc.json", "w"))
    np.savez(RES / "posthoc_cal.npz", **{f"{e}_{n}": v for e, (M, kk, L) in cal.items()
                                         for n, v in [("M", M), ("knots", kk), ("luts", L)]})


if __name__ == "__main__":
    main()
