# Explicit adapter vs latent predictor (ImageNet-ES-Diverse)

**Question.** The QAE model (`testbed/qae/model_text.py`) uses a JEPA-style latent predictor
(`testbed/qa_scoring/jepa_qa.py: Predictor`, ~8M params) to map the auto-exposure (AE) shot's DINOv2
tokens plus an action (source -> target camera setting) to the tokens of the capture at the target
setting. Can that transition instead be an **explicit physical image operation** whose parameter values
come from a small MLP, applied to the AE image before the frozen encoder?

```
x_hat = F( clip( gain * grid * F^-1(x_AE) + offset [+ noise] ) )      (gain, offset, grid, noise) = MLP(action, stats(x_AE))
tokens_hat = DINOv2(x_hat)  ->  same selection head as for real captures
```

Only the parameter values are learned (~20k params); the form of the operation is fixed physics.
QAE raw data is not available here, so this is run on ImageNet-ES-Diverse (6 lighting envs x 1000
images x 27 settings of aperture x shutter x ISO, plus AE shots, all on nesl).

## Setup
- Split: the July study's seed-0 3/1/1 split by reference image (train 3600 / val 1200 / test 1200 scenes).
  Hyperparameters are chosen on val; test is reported once at the end.
- Source: the AE shot `ae_1` (its setting is unknown: ES-Diverse has no EXIF). Training pairs use the AE
  shot as source half the time and a manual capture (known setting) otherwise, as in `model_text.py`.
- Encoder: frozen `facebook/dinov2-small`, 224 x 168 input, layer-normed 12 x 16 patch tokens.
- Labels: ResNet-50 and OpenCLIP top-1 correctness of every capture (`downstream_eval/image_results.jsonl`).

## Transition models
| name | what it is |
|---|---|
| `copy` | AE tokens for every setting (lower bound) |
| `physics` | Mohammadali's `run_phys_es.py` v2: fitted tone LUT + one AE exposure per env (**uses the env label = oracle**) |
| `latent` | `jepa_qa.Predictor`, same width/depth, this grid and a 9-d action |
| `explicit_*` | `models.ExplicitAdapter`: `phys` gain = 2^(EV_tgt - EV_src) x residual, EV of the AE shot estimated by the MLP; `free` = gain straight from the MLP; options: source-image stats, low-res gain grid, learned tone curve, noise |

## Metrics
- Fidelity (AE source -> each of 27 settings): token cosine, pooled cosine, skill = 1 - MSE / MSE(copy),
  PSNR for image-space models; overall, per brightness band, and on held-out settings.
- Selection: a quality head (`AttnHead`, the `TextHead` design with a learned query) trained on REAL
  tokens of all settings is applied to the predicted tokens; pick the argmax setting, score the real
  capture there. Baselines: AE, best fixed setting (train), `direct` (head on AE tokens + setting),
  `q_real` (head on real tokens of all 27 = needs all captures), oracle.

## Run (nesl, `qae` env)
```
cd explicit_adapter
~/miniforge3/envs/qae/bin/python run.py specs/round1.json --gpu 1 --only heads_s0,latent_s0
```
Outputs go to `_out/<spec name>/` (gitignored): `result.json`, `metrics_<split>.npz`, `model.pt`.
