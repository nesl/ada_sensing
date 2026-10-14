# Exposure Emulation Pipeline — Design Sketch

Goal: a software emulator that takes any source footage and renders what a
VIRTUAL camera would have captured at an arbitrary config
`(shutter t, ISO g, aperture A, scene brightness EV)` — continuously, at
scale. No physical capture rig: the camera model is parametric, and its
radiometric components are anchored in public RAW datasets (which are linear
sensor data, no tone-curve reversal needed).

## Pipeline

```
   PRIMARY source: RAW (ROD/MultiRAW/NOD —      FALLBACK source: 8-bit sRGB
   10-24 bit linear Bayer, known exposure)      video (H.264, unknown AE/ISP)
                                   │                        │
  [1] To linear RGB                │  RAW: black-level      │  8-bit: undo
                                   │  subtract, demosaic,   │  gamma/tone curve
                                   │  normalize — EXACT,    │  L = f_src⁻¹(I) —
                                   │  no reversal needed    │  APPROXIMATE
                                   ▼◄───────────────────────┘
  [2] Exposure scaling             │  r = (t/t_ref)·(g/g_ref)·(A_ref/A)²·2^EV
                                   │  L' = r · L        (shutter, ISO-as-gain,
                                   │  aperture area, scene brightness — one
                                   ▼  multiplicative product in linear light)
  [3] Motion blur synthesis        │  slow shutters (t > source frame time):
                                   │  L'' = weighted average of the k source
                                   │  frames the exposure window spans
                                   ▼
  [4] Sensor saturation            │  clip: L''' = min(L'', 1.0)
                                   │  (emulated full-well; blown = blown)
                                   ▼
  [5] Noise injection              │  shot noise ~ Poisson(L'''/g·K)·g/K
                                   │  read noise ~ N(0, σ_r(g))
                                   │  params (K, σ_r) calibrated PER ISO
                                   ▼
  [6] ISP simulation               │  chosen tone curve f_cam, quantize to
                                   │  8 bit, optional denoise/sharpen for a
                                   │  realistic processed output
                                   ▼
                     emulated capture at config (t, g, A, EV)
```

## Calibration & validation (RAW-anchored, no physical rig)

- **Noise model `(K, σ_r)` per ISO**: take published per-sensor calibrations
  (ELD-style affine shot+read models) or fit directly from dark/flat regions
  of RAW datasets (ROD, MultiRAW, NOD) — RAW noise statistics are directly
  measurable, no reversal involved.
- **Tone curves**: `f_src⁻¹` approximated as standard sRGB/BT.709 gamma for
  8-bit sources (acknowledged approximation); `f_cam` is ours to choose —
  a fixed, documented curve keeps the virtual camera reproducible.
- **Validate radiometry on RAW data**: render a RAW frame at exposure A,
  emulate config B from it, compare against the same scene actually captured
  at B where datasets provide multi-exposure or paired data; additionally
  check that downstream task scores (detection/classification) on emulated
  configs rank consistently with real captures at those configs.
- The 8-bit-source path (BDD video) inherits stage-1 caveats; quantify its
  divergence by running the same emulation from a RAW source downconverted
  to 8-bit and measuring the gap.

## Caveats — what emulation CANNOT do (each tied to its stage)

| Stage | Hard limit | Consequence |
|---|---|---|
| [1] | 8-bit sources (e.g. BDD) are tone-mapped H.264, already shot through another camera's AE + ISP | We emulate "camera pointed at a camera's output," not the scene; bounds everything below. Vanishes when the source is RAW (ROD, MultiRAW, NOD). |
| [1] | Highlights clipped in the SOURCE are unrecoverable | "Darker" renders of blown regions turn flat gray, not into true sky/sun detail a real short exposure would see. |
| [1] | Shadows near 0 carry 8-bit quantization + codec artifacts, not scene detail | "Brighter" renders amplify banding/blocking; limits how far up the emulator can push before diverging from reality. |
| [2] | ISO scaling ≠ pure gain on real sensors (analog→digital gain transitions, sensor-specific behavior) | Noise model must be calibrated per ISO, not extrapolated as a single formula. |
| [3] | Shutters FASTER than the source's (unknown, baked-in) shutter cannot be emulated | Cannot invent sharpness. Frame-averaging only emulates blur ≥ 1 source frame time (≥ ~1/30 s at 30 fps); the 1/30 s → source-shutter range is approximated as "no added blur." |
| [3] | Frame-averaging is a piecewise-constant approximation of continuous blur | Fast-moving objects show step artifacts vs the smooth streaks of a real slow shutter. |
| [4] | Saturation is modeled at the emulated sensor only | Interaction with source clipping (stage 1) compounds: regions clipped in both are doubly unfaithful. |
| [5] | Injected noise is only as real as its calibration | Spatially-correlated noise, fixed-pattern noise, and ISP denoising residue are hard to match; validate against real captures per ISO. |
| [6] | Real ISPs (multi-frame merge, adaptive denoise, local tone mapping) are proprietary and scene-adaptive | Our virtual ISP is a fixed, documented approximation — a feature for reproducibility, but emulated output won't match any specific commercial camera's look. |
| — | Aperture: brightness term only; depth-of-field NOT modeled | Fine for driving scenes (deep DoF beyond ~10 m); would need per-frame depth (bokeh-rendering literature) if shallow DoF ever matters. |
| — | Better sources shrink most stage-1 caveats | RAW / HDR / high-bit-depth source video (e.g., HDR datasets, or RAW captures from the phone itself) directly widens the trustworthy emulation range. |

## Relation to prior work

- Shutter emulation from brackets + offline AE benchmarking: BorealHDR /
  "Exposing the Unseen" (IROS 2024) — our stage [2] follows their formulation;
  they validate ≤1.78% RMSE for static scenes from RAW brackets.
- ISO/noise synthesis: physics-based calibrated shot+read models (ELD-style,
  affine noise model), standard in the learned-denoising literature.
- Aperture/bokeh rendering: RealBokeh/Bokehlicious, MODEST, EBB!, DoF-NeRF —
  deliberately out of scope here (DoF negligible for our scenes).
- RAW source datasets: ROD/RAOD (24-bit HDR driving, detection labels),
  MultiRAW (5 sensors, 10-24 bit), NOD (14-bit night), PASCALRAW (12-bit day).
