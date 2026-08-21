"""Generate the gaplike logo.

The mark is not drawn by hand: it is the package's own object. A chirp is
sampled, multiplied by a Planck-tapered gate, and emitted as one SVG path per
surviving stretch. Three things follow, and all three are the point of the
package:

* the stroke rises out of the time axis and settles back onto it, because the
  record edges are tapered rather than cut square;
* it stops dead in the middle, because that stretch of data does not exist;
* it resumes with the amplitude climbing back through a short taper.

A hairline axis runs the full width, so the eye reads "the clock keeps
running, the data do not".

    python make_logo.py
"""
from __future__ import annotations

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

INK_LIGHT = "#1f2933"          # ink for light backgrounds
INK_DARK = "#e8ecf1"           # ink for dark backgrounds
NS = 2400                      # samples along the curve

# the gated chirp, tuned so the mark stays legible down to ~16 px
GAPS = [(0.56, 0.080)]         # (centre, half-width), as a fraction of the record
TAPER = 0.035                  # Planck taper on the gap edges
EDGE = 0.14                    # Planck taper on the record edges
CHIRP = dict(cycles0=1.7, cycles_rise=1.0, grow=0.45)


# --------------------------------------------------------------------------
# the signal
# --------------------------------------------------------------------------
def planck(x, w):
    """C-infinity ramp from 0 to 1 over ``[0, w]``; cf. `gaplike.gaps.planck_edge`."""
    x = np.clip(np.asarray(x, float) / w, 1e-6, 1 - 1e-6)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(1.0 / x - 1.0 / (1.0 - x)))


def gate(t, gaps=GAPS, taper=TAPER, edge=EDGE):
    """Zero inside each gap, Planck-tapered at the gap and record edges."""
    g = np.ones_like(t)
    for centre, half in gaps:
        d = np.abs(t - centre)
        k = np.ones_like(t)
        k[d <= half] = 0.0
        ramp = (d > half) & (d < half + taper)
        k[ramp] = planck(d[ramp] - half, taper)
        g *= k
    if edge > 0:
        g *= planck(np.clip(t, 0, edge), edge) * planck(np.clip(1 - t, 0, edge), edge)
    return g


def chirp(t, cycles0=1.7, cycles_rise=1.0, grow=0.45):
    """An inspiral over ``t`` in [0, 1]: frequency and amplitude both rise."""
    env = (1.0 - grow) + grow * t
    return env * np.sin(2 * np.pi * (cycles0 * t + cycles_rise * t * t))


def _segments(xs, ys, g, eps=1e-4):
    """Split the curve wherever the gate vanishes."""
    live = g > eps
    out, i = [], 0
    while i < live.size:
        if not live[i]:
            i += 1
            continue
        j = i
        while j < live.size and live[j]:
            j += 1
        if j - i > 2:
            out.append((xs[i:j], ys[i:j]))
        i = j
    return out


def _to_path(xs, ys):
    return (f"M{xs[0]:.2f},{ys[0]:.2f}"
            + "".join(f"L{x:.2f},{y:.2f}" for x, y in zip(xs[1:], ys[1:])))


def wave_paths(x0, x1, ycen, amp):
    """SVG path strings for the gated chirp drawn across ``[x0, x1]``."""
    t = np.linspace(0.0, 1.0, NS)
    g = gate(t)
    y = chirp(t, **CHIRP) * g
    y = y / np.abs(y).max()
    return [_to_path(a, b) for a, b in
            _segments(x0 + t * (x1 - x0), ycen - y * amp, g)]


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def _wave_svg(x0, x1, ycen, amp, ink, stroke, axis_pad=2.0):
    body = "".join(
        f'<path d="{d}" fill="none" stroke="{ink}" stroke-width="{stroke:.2f}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>' for d in
        wave_paths(x0, x1, ycen, amp))
    axis = (f'<line x1="{x0 - axis_pad:.1f}" y1="{ycen:.1f}" '
            f'x2="{x1 + axis_pad:.1f}" y2="{ycen:.1f}" stroke="{ink}" '
            f'stroke-width="{stroke * 0.44:.2f}" stroke-linecap="round" '
            f'opacity="0.32"/>')
    return axis + body


def mark(ink=INK_LIGHT, size=128, stroke=3.2, ring=False):
    """Square mark. ``ring=True`` adds a hairline circle, for avatars."""
    c = size / 2
    pad = 15 if ring else 8
    circ = (f'<circle cx="{c}" cy="{c}" r="{c - 4}" fill="none" stroke="{ink}" '
            f'stroke-width="{stroke * 0.55:.2f}" opacity="0.5"/>' if ring else "")
    inner = _wave_svg(pad, size - pad, c, size * 0.33, ink, stroke)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}" role="img" aria-label="gaplike">'
            f'{circ}{inner}</svg>')


def _text_path(text, px, font, weight="normal"):
    """Type as outlines, so the wordmark renders identically everywhere."""
    tp = TextPath((0, 0), text, size=px,
                  prop=FontProperties(family=font, weight=weight))
    d = []
    for v, code in tp.iter_segments():
        if code == 1:
            d.append(f"M{v[0]:.2f},{-v[1]:.2f}")
        elif code == 2:
            d.append(f"L{v[0]:.2f},{-v[1]:.2f}")
        elif code == 3:
            d.append(f"Q{v[0]:.2f},{-v[1]:.2f},{v[2]:.2f},{-v[3]:.2f}")
        elif code == 4:
            d.append(f"C{v[0]:.2f},{-v[1]:.2f},{v[2]:.2f},{-v[3]:.2f},"
                     f"{v[4]:.2f},{-v[5]:.2f}")
        elif code == 79:
            d.append("Z")
    return "".join(d), tp.get_extents()


def wordmark(ink=INK_LIGHT, h=148, font=None, px=64, gapx=26):
    """Mark on the left, the name set as outlines on the right."""
    # outlines, so the file renders identically without the font installed
    font = font or ["FreeSans", "Helvetica", "Arial", "DejaVu Sans"]
    ycen = h / 2
    wave_x0, wave_x1 = 12, 188
    d_text, ext = _text_path("gaplike", px, font)
    tx = wave_x1 + gapx
    w = int(tx + ext.width + 14)
    inner = _wave_svg(wave_x0, wave_x1, ycen, h * 0.30, ink, 3.5)
    text = (f'<g transform="translate({tx:.1f},{ycen + px * 0.36:.1f})">'
            f'<path d="{d_text}" fill="{ink}"/></g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="gaplike">'
            f'{inner}{text}</svg>')


if __name__ == "__main__":
    out = {
        "logo.svg": mark(INK_LIGHT),
        "logo-dark.svg": mark(INK_DARK),
        "logo-ring.svg": mark(INK_LIGHT, ring=True),
        "logo-wordmark.svg": wordmark(INK_LIGHT),
        "logo-wordmark-dark.svg": wordmark(INK_DARK),
        # favicon: heavier stroke, tighter padding -- legible at 16 px
        "favicon.svg": mark(INK_LIGHT, size=64, stroke=5.0),
    }
    for name, svg in out.items():
        with open(name, "w") as fh:
            fh.write(svg + "\n")
        print(f"wrote {name}  ({len(svg) / 1024:.1f} kB)")
