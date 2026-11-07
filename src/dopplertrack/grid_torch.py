"""torch port of tracks.grid_search: every (v, d) grid pair scored in batched tensors.

Same inputs, outputs and maths as the numpy version (ideal-curve interpolation, sweep weights,
bilinear sampling with out-of-grid = 0, ridge contrast, first-maximum tie-breaking); float32 on
the device. Times are shifted to the spectrogram's first frame before the float32 cast so a
many-hour recording keeps sub-millisecond resolution.
"""
from __future__ import annotations

import numpy as np
import torch

from .audio import Spectrogram

SPAN_S = 400.0      # tracks._shape defaults
SPAN_N = 161
ELEMENTS_PER_BATCH = 24_000_000   # (v, d) pairs x T0 x N x F0 per batch; ~2 GB of temporaries


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _interp_rows(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """np.interp(x, xp[b], fp[b], left=nan, right=nan) for each row b. x (M,), xp/fp (B, K)."""
    xb = x.expand(xp.shape[0], -1).contiguous()
    i = torch.searchsorted(xp, xb).clamp(1, xp.shape[1] - 1)
    x0, x1 = xp.gather(1, i - 1), xp.gather(1, i)
    y0, y1 = fp.gather(1, i - 1), fp.gather(1, i)
    y = y0 + (y1 - y0) * (xb - x0) / (x1 - x0)
    outside = (xb < xp[:, :1]) | (xb > xp[:, -1:])
    return y.masked_fill(outside, float("nan"))


class _Grid:
    """The spectrogram on the device plus bilinear sampling identical to Spectrogram.sample."""

    def __init__(self, sp: Spectrogram, device: str, fat_bins: int = 1):
        self.nt, self.nf = len(sp.t), len(sp.f)
        self.dt, self.df = sp.dt, sp.df
        self.f_first = float(sp.f[0])
        self.S = torch.as_tensor(np.ascontiguousarray(sp.S), dtype=torch.float32, device=device)[None, None]
        if fat_bins > 1:
            # Max over `fat_bins` frequency bins (centred). The -inf padding of max_pool gives the
            # same maxima as a reflected edge, whose mirrored bins are already inside the window.
            k = fat_bins // 2 * 2 + 1
            self.S = torch.nn.functional.max_pool2d(self.S, (k, 1), stride=1, padding=(k // 2, 0))

    def sample(self, ti: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """ti: fractional frame index broadcastable to f; f: frequencies in Hz. Off-grid / NaN -> 0.

        grid_sample (align_corners=True) is exact bilinear interpolation on the grid; points
        outside [0, n-1] on either axis are zeroed explicitly, as Spectrogram.sample does,
        instead of grid_sample's partial blend with the zero padding."""
        fi = (f - self.f_first) / self.df
        ti = ti.expand_as(fi)
        ok = (ti >= 0) & (ti <= self.nt - 1) & (fi >= 0) & (fi <= self.nf - 1)   # NaN compares False
        x = ti * (2.0 / max(self.nt - 1, 1)) - 1.0
        y = fi * (2.0 / max(self.nf - 1, 1)) - 1.0
        grid = torch.stack([x, y], dim=-1).nan_to_num(nan=-2.0).reshape(1, 1, -1, 2)
        v = torch.nn.functional.grid_sample(self.S, grid, mode="bilinear", padding_mode="zeros",
                                            align_corners=True).reshape(fi.shape)
        return torch.where(ok, v, 0.0)


def grid_search_torch(sp: Spectrogram, t_start, t_end, f_lo, f_hi, c, harmonics, v_grid, d_grid,
                      f0_step, t0_step, n_curve, t0_range, f0_range, per_t0, contrast: bool, fat_bins: int,
                      contrast_df: float, sweep_floor: float, device: str):
    """See tracks.grid_search."""
    origin = float(sp.t[0])
    f0s_np = np.arange(*(f0_range or (f_lo * 1.15, f_hi * 0.85)), f0_step)
    t0s_np = np.arange(*(t0_range or (t_start, t_end)), t0_step)
    tq_np = np.linspace(t_start, t_end, n_curve)
    T0, N, F0 = len(t0s_np), len(tq_np), len(f0s_np)

    as_t = lambda a: torch.as_tensor(np.asarray(a, np.float64), dtype=torch.float32, device=device)
    grid = _Grid(sp, device, fat_bins if contrast and fat_bins else 1)
    f0s = as_t(f0s_np)
    tq_idx = as_t((tq_np - origin) / sp.dt)                          # fractional frame of each curve sample
    rel = as_t((tq_np[None, :] - t0s_np[:, None]).ravel())          # (T0*N,) obs time relative to emission CPA
    dq = float(tq_np[1] - tq_np[0]) if N > 1 else 1.0
    tau = torch.linspace(-SPAN_S, SPAN_S, SPAN_N, device=device, dtype=torch.float32)

    vv, dd = np.meshgrid(np.asarray(v_grid, float), np.asarray(d_grid, float), indexing="ij")   # v outer, d inner
    V_all, D_all = as_t(vv.ravel()), as_t(dd.ravel())
    batch = max(1, ELEMENTS_PER_BATCH // max(T0 * N * F0, 1))

    best_s = torch.full((T0,), -float("inf"), device=device)
    best_p = torch.zeros((T0, 3), device=device)
    for b0 in range(0, len(V_all), batch):
        V = V_all[b0:b0 + batch, None]; D = D_all[b0:b0 + batch, None]     # (B, 1)
        r = torch.sqrt(D * D + (V * tau) ** 2)
        dto = tau + r / c                                                  # (B, K), increasing
        ratio = c / (c + V * V * tau / r)
        rat = _interp_rows(rel, dto, ratio).reshape(-1, T0, N)             # (B, T0, N)
        nan = torch.isnan(rat)
        if contrast:
            g = torch.gradient(rat, spacing=dq, dim=2)[0].abs().nan_to_num(nan=0.0)
            w = g / g.amax(dim=2, keepdim=True).clamp_min(1e-12) + sweep_floor
            w = w.masked_fill(nan, 0.0)
        else:
            w = torch.ones_like(rat)
        wsum = w.sum(dim=2).clamp_min(1e-12)                               # (B, T0)
        total = torch.zeros((rat.shape[0], T0, F0), device=device)
        ti = tq_idx[None, None, :, None]                                   # (1, 1, N, 1)
        for h in range(1, harmonics + 1):
            F = h * f0s * rat[..., None]                                   # (B, T0, N, F0)
            if contrast:
                vals = grid.sample(ti, F) - 0.5 * (grid.sample(ti, F * (1 + contrast_df)) + grid.sample(ti, F * (1 - contrast_df)))
            else:
                vals = grid.sample(ti, F)
            total += torch.einsum("btn,btnf->btf", w, vals) / wsum[..., None]
        total /= harmonics
        sc, j = total.max(dim=2)                                           # (B, T0) best f0 per (v, d, t0)
        bsc, bk = sc.max(dim=0)                                            # best (v, d) per t0 in this batch
        upd = bsc > best_s                                                 # strict: earlier (v, d) wins ties
        best_s = torch.where(upd, bsc, best_s)
        k_glob = bk + b0
        cand = torch.stack([f0s[j.gather(0, bk[None])[0]], V_all[k_glob], D_all[k_glob]], dim=1)
        best_p = torch.where(upd[:, None], cand, best_p)

    bs = best_s.cpu().numpy().astype(np.float64)
    bp = best_p.cpu().numpy().astype(np.float64)
    if per_t0:
        return t0s_np, bs, bp
    i = int(np.argmax(bs))
    return float(bs[i]), (float(bp[i, 0]), float(bp[i, 1]), float(bp[i, 2]), float(t0s_np[i]))
