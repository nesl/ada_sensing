"""Transition models (source shot + action -> the capture at the target setting) and selection heads."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ADIM, CONFIGS, D, ENVS, EV, GH, GW, H, W


# ---------------------------------------------------------------- explicit image operations
def srgb_to_linear(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def interp(x, xp, fp):
    """Piecewise-linear interpolation of x on (xp, fp), xp non-decreasing (K,); differentiable in x, xp, fp.
    Table lookups go through F.embedding: its backward sorts the indices, whereas plain indexing
    accumulates millions of pixel gradients into K slots with atomics (6.5x slower, same result)."""
    idx = torch.searchsorted(xp.detach().contiguous(), x.detach().contiguous(), right=True)
    idx = idx.clamp(1, xp.numel() - 1)
    tab = torch.stack([xp, fp], 1)
    lo, hi = F.embedding(idx - 1, tab), F.embedding(idx, tab)
    x0, y0, x1, y1 = lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1]
    w = ((x - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0, 1)
    return y0 + w * (y1 - y0)


class ToneCurve(nn.Module):
    """Per-channel monotone tone curve F: linear [0,1] -> display [0,1] on sqrt-spaced knots
    (same knots as exposure_sim/run_phys_es.py), initialised to sRGB; F(0)=0, F(1)=1 and
    strictly increasing by construction."""

    def __init__(self, learn=True, n_knots=33):
        super().__init__()
        self.register_buffer("knots", torch.linspace(0, 1, n_knots) ** 2)
        init = linear_to_srgb(self.knots).repeat(3, 1)
        inc = torch.diff(init, dim=1).clamp(min=1e-4)
        inc = inc / inc.sum(1, keepdim=True) * (n_knots - 1)
        self.raw = nn.Parameter(torch.log(torch.expm1(inc)), requires_grad=learn)

    def values(self):
        inc = F.softplus(self.raw)
        v = torch.cumsum(inc, 1) / inc.sum(1, keepdim=True)
        return torch.cat([torch.zeros_like(v[:, :1]), v], 1)          # (3, K)

    def fwd(self, lin):                                              # (B,3,H,W)
        v = self.values()
        return torch.stack([interp(lin[:, c], self.knots, v[c]) for c in range(3)], 1)

    def inv(self, disp):
        v = self.values()
        return torch.stack([interp(disp[:, c], v[c], self.knots) for c in range(3)], 1)


STATS = 22


def image_stats(x):
    """Exposure cues of the source image: per-channel linear mean, log2 linear luminance,
    16-bin display-luminance histogram, clipped and black pixel fractions."""
    lin = srgb_to_linear(x)
    w = x.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    lum = (x * w).sum(1)
    llin = (lin * w).sum(1).mean((1, 2))
    hist = F.one_hot((lum * 16).long().clamp(0, 15), 16).float().mean((1, 2))
    mx = x.amax(1)
    return torch.cat([4 * lin.mean((2, 3)), (torch.log2(llin + 1e-4) / 8)[:, None], hist,
                      (mx > 0.98).float().mean((1, 2))[:, None], (mx < 0.02).float().mean((1, 2))[:, None]], 1)


class ExplicitAdapter(nn.Module):
    """x_hat = F( clip( gain * grid * F^-1(x) + offset [+ noise] ) ): an explicit exposure operation
    on the source image; the parameter VALUES come from an MLP of (action, source-image stats).

    gain='phys': 2^(EV_tgt - EV_src) * exp(residual), EV_src known for a manual source and
                 estimated by the MLP for the AE shot (whose setting is unknown);
    gain='free': exp(MLP output), no physics prior.
    grid=(gh, gw): spatially varying log-gain on a low-res grid (local ISP / vignetting); (0, 0) = off.
    """

    def __init__(self, gain="phys", use_stats=True, grid=(0, 0), learn_tone=True, noise=False, hidden=128):
        super().__init__()
        self.gain, self.use_stats, self.grid, self.noise = gain, use_stats, tuple(grid), noise
        self.n_grid = self.grid[0] * self.grid[1]
        self.tone = ToneCurve(learn=learn_tone)
        n_out = 3 + 3 + 1 + self.n_grid + (2 if noise else 0)
        self.mlp = nn.Sequential(nn.Linear(ADIM + (STATS if use_stats else 0), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.ev_ae0 = nn.Parameter(torch.tensor(float(EV.mean())))   # prior for the AE shot's log2 exposure

    def params(self, x, a, ev_src, ev_tgt):
        z = self.mlp(torch.cat([a, image_stats(x)], 1) if self.use_stats else a)
        if self.gain == "phys":
            ev_hat = torch.where(torch.isnan(ev_src), self.ev_ae0 + 4 * z[:, 6], ev_src)
            log_gain = math.log(2) * (ev_tgt - ev_hat)[:, None] + z[:, 0:3]
        else:
            ev_hat = None
            log_gain = 4 * z[:, 0:3]
        off = 0.02 * z[:, 3:6]
        o = 7
        grid = None
        if self.n_grid:
            g = 0.5 * torch.tanh(z[:, o:o + self.n_grid]).view(-1, 1, *self.grid)
            grid = torch.exp(F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False))
            o += self.n_grid
        noise = 0.02 * F.softplus(z[:, o:o + 2] - 2) if self.noise else None
        return log_gain, off, grid, noise, ev_hat

    def forward(self, x, a, ev_src, ev_tgt, stochastic=None):
        log_gain, off, grid, noise, _ = self.params(x, a, ev_src, ev_tgt)
        y = self.tone.inv(x) * torch.exp(log_gain)[:, :, None, None]
        if grid is not None:
            y = y * grid
        y = y + off[:, :, None, None]
        if noise is not None and (self.training if stochastic is None else stochastic):
            var = y.clamp(min=0) * noise[:, 0:1, None, None] + noise[:, 1:2, None, None] ** 2
            y = y + torch.randn_like(y) * var.sqrt()
        return self.tone.fwd(y.clamp(0, 1))


class PhysicsOracle:
    """exposure_sim/run_phys_es.py v2, training-free: Mohammadali's fitted per-channel tone LUT and one
    AE exposure per lighting env (e_fit). Uses the env label, i.e. ORACLE information about the AE
    exposure. x_hat = F(clip(F^-1(AE) * E_tgt / E_AE(env)))."""

    def __init__(self, repo_root, device):
        res = Path(repo_root) / "exposure_sim" / "results"
        self.luts = torch.tensor(np.load(res / "phys_tonecurve.npy"), dtype=torch.float32, device=device)
        self.knots = (torch.linspace(0, 1, self.luts.shape[1]) ** 2).to(device)
        e = json.load(open(res / "metrics_phys.json"))["e_fit"]
        self.e_ae = torch.tensor([e[env] for env in ENVS], dtype=torch.float32, device=device)
        self.E = torch.tensor([t * iso / n ** 2 for t, iso, n in CONFIGS], dtype=torch.float32, device=device)

    def __call__(self, x, env, tgt):
        lin = torch.stack([interp(x[:, c], self.luts[c], self.knots) for c in range(3)], 1)
        y = (lin * (self.E[tgt] / self.e_ae[env])[:, None, None, None]).clamp(0, 1)
        return torch.stack([interp(y[:, c], self.knots, self.luts[c]) for c in range(3)], 1)


# ---------------------------------------------------------------- latent baseline
class LatentPredictor(nn.Module):
    """jepa_qa.Predictor with this patch grid and action size: transformer over the source tokens plus
    an action embedding added to every token; residual output (predicts the change)."""

    def __init__(self, dim=D, width=384, depth=4, heads=6):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.zeros(1, GH * GW, width))
        self.act = nn.Sequential(nn.Linear(ADIM, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True, norm_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, dim)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, src, a):              # src (B, N, D) layer-normed tokens, a (B, ADIM)
        return src + self.out(self.tr(self.inp(src) + self.pos + self.act(a)[:, None]))


# ---------------------------------------------------------------- selection heads
class AttnHead(nn.Module):
    """testbed/qae/model_text.py TextHead with a learned query instead of a text query: the query
    (plus an optional setting embedding) cross-attends over patch tokens -> one logit per judge."""

    def __init__(self, dim=D, width=256, heads=4, layers=2, adim=0, nout=2):
        super().__init__()
        self.tok = nn.Linear(dim, width)
        self.pos = nn.Parameter(torch.zeros(1, GH * GW, width))
        self.q0 = nn.Parameter(torch.zeros(1, 1, width))
        self.qa = nn.Linear(adim, width) if adim else None
        self.attn = nn.ModuleList([nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(),
                                                nn.Linear(width * 2, width)) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, nout))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.q0, std=0.02)

    def forward(self, toks, a=None):
        kv = self.tok(toks) + self.pos
        q = self.q0.expand(toks.shape[0], -1, -1)
        if self.qa is not None:
            q = q + self.qa(a)[:, None]
        for at, m in zip(self.attn, self.mlp):
            q = q + at(q, kv, kv)[0]
            q = q + m(q)
        return self.out(q)[:, 0]
