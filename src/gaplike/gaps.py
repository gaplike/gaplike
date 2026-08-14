"""Gap patterns, gates and effective windows.

Conventions
-----------
* A **mask** is a boolean array of length ``n``: ``True`` = sample observed,
  ``False`` = sample lost to a gap.
* A **gate** is a float array in ``[0, 1]``: ``0`` inside gaps, with optionally
  tapered (Planck) edges.
* The **effective window** is ``gate * segment_window``; it multiplies the
  time series before any Fourier transform.  The exact time-domain likelihood
  (:class:`gaplike.likelihood.TimeDomainExact`) uses the *mask* only — no
  window ever appears there.

All builders accept arbitrary patterns: explicit intervals, periodic combs,
random (Poisson) gaps, or any user-supplied boolean mask.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import windows as _windows

__all__ = [
    "mask_from_intervals", "periodic_mask", "random_mask", "mask_from_gate",
    "duty_cycle", "window_power", "gap_intervals",
    "planck_edge", "gate_from_gaps", "gate_from_mask",
    "segment_window", "effective_window",
]


# --------------------------------------------------------------------------
# masks (True = observed)
# --------------------------------------------------------------------------
def mask_from_intervals(n, dt, gaps):
    """Boolean mask with the samples inside each ``(t_start, t_end)`` interval
    (seconds, gap = removed data) set to ``False``."""
    t = np.arange(n) * dt
    m = np.ones(n, bool)
    for t0, t1 in gaps:
        m &= ~((t >= t0) & (t < t1))
    return m


def periodic_mask(n, gap_samples, period_samples, offset=0):
    """Rectangular comb: ``gap_samples`` consecutive samples removed at the
    start of every ``period_samples`` block (the paper's 'drastic gaps')."""
    m = np.ones(n, bool)
    for s0 in range(offset, n, period_samples):
        m[s0:s0 + gap_samples] = False
    return m


def random_mask(n, dt, rate_per_day, duration_s, duration_jitter_s=0.0,
                rng=None):
    """Poisson-distributed gaps: ``rate_per_day`` expected gaps per day, each
    of ``duration_s`` seconds (Gaussian-jittered by ``duration_jitter_s``)."""
    rng = np.random.default_rng(rng)
    t_total = n * dt
    n_gaps = rng.poisson(rate_per_day * t_total / 86400.0)
    starts = rng.uniform(0.0, t_total, n_gaps)
    durs = np.maximum(duration_s
                      + duration_jitter_s * rng.standard_normal(n_gaps), dt)
    return mask_from_intervals(n, dt, zip(starts, starts + durs))


def mask_from_gate(gate):
    """Boolean mask of the strictly-zero samples of a gate."""
    return np.asarray(gate) > 0.0


def duty_cycle(mask):
    """Fraction of observed samples."""
    return float(np.mean(np.asarray(mask, dtype=float) > 0.0))


def window_power(w):
    """Mean squared window, ``W_2 = sum(w^2)/n`` — the normalizing constant of
    the scaled-Whittle ('normalizing constant') approximation."""
    w = np.asarray(w, float)
    return float(np.sum(w**2) / w.size)


def gap_intervals(mask, dt):
    """List of gap intervals ``(t_start, t_end)`` in seconds from a mask."""
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[True], m, [True]]).astype(int))
    starts = np.where(d == -1)[0]
    ends = np.where(d == +1)[0]
    return [(s * dt, e * dt) for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------
# gates (tapered or rectangular)
# --------------------------------------------------------------------------
def planck_edge(x, taper):
    """C-infinity ramp 0 -> 1 over ``[0, taper]``."""
    x = np.clip(np.asarray(x, float) / taper, 1e-6, 1 - 1e-6)
    with np.errstate(over="ignore"):        # exp overflow -> ramp saturates at 0
        return 1.0 / (1.0 + np.exp(1.0 / x - 1.0 / (1.0 - x)))


def gate_from_gaps(gaps, taper, n=None, dt=None, t=None):
    """Gate with gaps given as ``(t_center, duration)`` pairs and Planck-tapered
    edges extending over ``taper`` OUTSIDE each gap.

    Times can be in any unit as long as ``gaps``, ``taper`` and the time grid
    agree: pass either ``n, dt`` (grid ``t_k = k*dt``) or an explicit ``t``.
    Overlapping tapers multiply.
    """
    if t is None:
        t = np.arange(n) * dt
    t = np.asarray(t, float)
    keep = np.ones(t.size)
    for c, d in gaps:
        x = np.abs(t - c)
        half = d / 2.0
        k = np.ones(t.size)
        k[x <= half] = 0.0
        e = (x > half) & (x < half + taper)
        if np.any(e):
            k[e] = planck_edge(x[e] - half, taper)
        keep *= k
    return keep


def gate_from_mask(mask, dt, taper_s=0.0):
    """Gate from an ARBITRARY boolean mask: 0 on removed samples, Planck ramp
    over ``taper_s`` seconds as a function of the distance to the nearest
    removed sample (``taper_s = 0`` gives the rectangular gate)."""
    m = np.asarray(mask, bool)
    g = m.astype(float)
    if taper_s > 0.0 and (~m).any() and m.any():
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(m) * dt
        ramp = dist < taper_s
        g[ramp & m] = planck_edge(dist[ramp & m], taper_s)
    return g


# --------------------------------------------------------------------------
# segment window / effective window
# --------------------------------------------------------------------------
def segment_window(n, alpha=0.05):
    """Tukey segment-edge taper (``alpha = 0`` gives all ones)."""
    return _windows.tukey(n, alpha)


def effective_window(gate, segment=None):
    """``w_eff = segment * gate`` (segment defaults to all ones)."""
    gate = np.asarray(gate, float)
    return gate if segment is None else np.asarray(segment, float) * gate
