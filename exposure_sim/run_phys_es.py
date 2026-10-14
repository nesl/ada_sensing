"""
Physical exposure simulator on the same ImageNet-ES-Diverse benchmark.

v1: pure physics -- assume sRGB gamma, scale linear by exposure ratio, clip.
v2: fitted physics -- iteratively fit the camera's per-channel tone curve F
    (monotone LUT, linear -> sRGB) from the 5 calibration images, so that
    real ~= F(r * F^-1(AE)) with r the physical exposure ratio. Per-env AE
    exposure refined through the model. Eval on the same 10 samples.

No networks, no pretrained weights. Fit uses calibration images only.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

import run_es_diverse as R

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
E_AE0 = json.load(open(RES / "metrics.json"))["e_ae"]

N_KNOTS = 33
KNOTS = (np.linspace(0, 1, N_KNOTS) ** 2)  # sqrt-spaced: shadow resolution


def imread(p, size=None):
    im = Image.open(p).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, np.float32) / 255.0


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


class ToneCurve:
    """Per-channel monotone LUT F: linear [0,1] -> sRGB [0,1], with inverse."""

    def __init__(self, luts=None):
        if luts is None:  # init = standard sRGB encoding
            luts = np.tile(linear_to_srgb(KNOTS), (3, 1))
        self.luts = luts

    def fwd(self, lin):
        return np.stack([np.interp(lin[..., c], KNOTS, self.luts[c]) for c in range(3)], -1)

    def inv(self, srgb):
        return np.stack([np.interp(srgb[..., c], self.luts[c], KNOTS) for c in range(3)], -1)


def collect_pairs(calib):
    """(ae_img, real_img, env, param_k) for all calib images/configs, once."""
    out = []
    for syn, stem in calib:
        for env in R.ENVS:
            ae_p = R.DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG"
            ae = imread(ae_p)
            for k in range(1, 28):
                rp = R.DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG"
                if rp.exists():
                    out.append((ae, imread(rp, (ae.shape[1], ae.shape[0])), env, k))
    return out


def fit_tone_curve(pairs, e_ae, n_iter=3):
    tc = ToneCurve()
    rng = np.random.default_rng(0)
    for it in range(n_iter):
        xs, ys = [], []
        for ae, real, env, k in pairs:
            t, iso, n = R.param_config(k)
            r = R.exposure_product(t, iso, n) / e_ae[env]
            pred_lin = np.clip(tc.inv(ae) * r, 0, 1)
            idx = rng.choice(pred_lin.shape[0] * pred_lin.shape[1], 4000, replace=False)
            xs.append(pred_lin.reshape(-1, 3)[idx])
            ys.append(real.reshape(-1, 3)[idx])
        X, Y = np.concatenate(xs), np.concatenate(ys)
        luts = np.zeros((3, N_KNOTS), np.float32)
        for c in range(3):
            v = np.full(N_KNOTS, np.nan)
            for i, kn in enumerate(KNOTS):
                lo = KNOTS[max(i - 1, 0)]
                hi = KNOTS[min(i + 1, N_KNOTS - 1)]
                m = (X[:, c] >= (kn + lo) / 2) & (X[:, c] <= (kn + hi) / 2)
                if m.sum() > 80:
                    v[i] = np.median(Y[m, c])
            ok = ~np.isnan(v)
            v = np.interp(KNOTS, KNOTS[ok], v[ok])
            v[0] = 0.0
            luts[c] = np.maximum.accumulate(v)
        tc = ToneCurve(luts)

        # refine per-env exposure through the fitted curve (mean matching, secant)
        for env in R.ENVS:
            sub = [(a, b, e, kk) for a, b, e, kk in pairs if e == env and 0.05 < b.mean() < 0.6]
            if not sub:
                continue

            def err(e_val):
                tot = 0.0
                for ae, real, _e, kk in sub:
                    t, iso, n = R.param_config(kk)
                    r = R.exposure_product(t, iso, n) / e_val
                    sim = tc.fwd(np.clip(tc.inv(ae) * r, 0, 1))
                    tot += np.log(max(sim.mean(), 1e-4)) - np.log(max(real.mean(), 1e-4))
                return tot / len(sub)

            x0, x1 = np.log(e_ae[env]), np.log(e_ae[env]) + 0.3
            f0, f1 = err(np.exp(x0)), err(np.exp(x1))
            for _ in range(3):
                if abs(f1 - f0) < 1e-4:
                    break
                x2 = np.clip(x1 - f1 * (x1 - x0) / (f1 - f0), x1 - 1, x1 + 1)
                x0, f0, x1 = x1, f1, x2
                f1 = err(np.exp(x1))
            e_ae[env] = float(np.exp(x1))
        print(f"iter {it}: E_AE {({k: round(v,3) for k,v in e_ae.items()})}", flush=True)
    return tc, e_ae


def psnr(a, b):
    m = ((a - b) ** 2).mean()
    return 10 * np.log10(1 / max(m, 1e-10))


def main():
    calib, samples = R.list_samples()
    pairs = collect_pairs(calib)
    print(f"{len(pairs)} calibration pairs")

    tc, e_fit = fit_tone_curve(pairs, dict(E_AE0))
    np.save(RES / "phys_tonecurve.npy", tc.luts)

    rows = []
    for si, (syn, stem) in enumerate(samples):
        sdirs = sorted(RES.glob(f"{si:02d}_*"))
        sdir = sdirs[0]
        for env in R.ENVS:
            ae = imread(R.DATA / "auto_exposure" / env / "param_1" / syn / f"{stem}.JPEG")
            lin_v1 = srgb_to_linear(ae)
            lin_v2 = tc.inv(ae)
            for k in range(1, 28):
                t, iso, n = R.param_config(k)
                E = R.exposure_product(t, iso, n)
                real = imread(R.DATA / "param_control" / env / f"param_{k}" / syn / f"{stem}.JPEG",
                              (ae.shape[1], ae.shape[0]))
                v1 = linear_to_srgb(np.clip(lin_v1 * E / E_AE0[env], 0, 1))
                v2 = tc.fwd(np.clip(lin_v2 * E / e_fit[env], 0, 1))
                Image.fromarray((v2 * 255).round().astype(np.uint8)).save(
                    sdir / env / f"phys_param_{k}.jpg", quality=92)
                rows.append({"sample": sdir.name, "env": env, "param": k,
                             "psnr_v1": psnr(v1, real), "psnr_v2": psnr(v2, real),
                             "real_mean": float(real.mean())})
        print(f"[{si+1}/{len(samples)}] {syn}/{stem}", flush=True)

    a = np.array([[x["psnr_v1"], x["psnr_v2"], x["real_mean"]] for x in rows])
    print(f"\noverall: phys-v1 {a[:,0].mean():.2f} dB | phys-v2 (fitted tone) {a[:,1].mean():.2f} dB")
    for lo, hi, tag in [(0, .08, "near-black"), (.08, .3, "dark-mid"), (.3, .6, "well-exposed"), (.6, 1, "bright/blown")]:
        m = (a[:, 2] >= lo) & (a[:, 2] < hi)
        if m.sum():
            print(f"  {tag:13s} n={int(m.sum()):4d}  v1 {a[m,0].mean():5.2f} | v2 {a[m,1].mean():5.2f}")
    json.dump({"e_fit": e_fit, "rows": rows}, open(RES / "metrics_phys.json", "w"))


if __name__ == "__main__":
    main()
