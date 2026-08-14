import numpy as np
import pytest

from gaplike import gaps


def test_mask_from_intervals():
    m = gaps.mask_from_intervals(10, 1.0, [(2.0, 4.0), (7.0, 8.0)])
    assert m.tolist() == [1, 1, 0, 0, 1, 1, 1, 0, 1, 1]


def test_periodic_mask():
    m = gaps.periodic_mask(10, 2, 5)
    assert m.tolist() == [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]
    assert gaps.duty_cycle(m) == 0.6


def test_random_mask_reproducible():
    m1 = gaps.random_mask(5000, 10.0, rate_per_day=40, duration_s=300.0, rng=3)
    m2 = gaps.random_mask(5000, 10.0, rate_per_day=40, duration_s=300.0, rng=3)
    assert np.array_equal(m1, m2)
    assert 0.3 < gaps.duty_cycle(m1) < 1.0


def test_gap_intervals_roundtrip():
    m = gaps.mask_from_intervals(100, 2.0, [(10.0, 20.0), (50.0, 54.0)])
    iv = gaps.gap_intervals(m, 2.0)
    assert iv == [(10.0, 20.0), (50.0, 54.0)]


def test_gate_from_gaps_matches_manual():
    n, dt = 1000, 1.0
    g = gaps.gate_from_gaps([(500.0, 100.0)], taper=50.0, n=n, dt=dt)
    t = np.arange(n) * dt
    inside = np.abs(t - 500.0) <= 50.0
    assert np.all(g[inside] == 0.0)
    far = np.abs(t - 500.0) >= 100.0
    assert np.all(g[far] == 1.0)
    edge = (np.abs(t - 500.0) > 50.0) & (np.abs(t - 500.0) < 100.0)
    # ramp values in (0, 1]; the C-inf ramp saturates to 1.0 in double
    # precision near its outer edge
    assert np.all((g[edge] > 0) & (g[edge] <= 1))
    assert 0.4 < g[np.argmin(np.abs(t - 575.0))] < 0.6  # midpoint of the ramp
    # non-decreasing ramp on the right edge
    right = np.where((t > 550.0) & (t < 600.0))[0]
    assert np.all(np.diff(g[right]) >= 0)


def test_gate_from_gaps_explicit_time_grid():
    t = np.arange(240) / 240.0          # e.g. hours
    g1 = gaps.gate_from_gaps([(0.5, 0.2)], taper=0.05, t=t)
    g2 = gaps.gate_from_gaps([(0.5 * 3600, 0.2 * 3600)], taper=0.05 * 3600,
                             n=240, dt=3600 / 240.0)
    assert np.allclose(g1, g2, atol=1e-12)


def test_gate_from_mask_rectangular_and_tapered():
    m = gaps.periodic_mask(100, 5, 25)
    g0 = gaps.gate_from_mask(m, dt=1.0)
    assert np.array_equal(g0, m.astype(float))
    g1 = gaps.gate_from_mask(m, dt=1.0, taper_s=4.0)
    assert np.all(g1[~m] == 0.0)
    assert np.all(g1 <= 1.0) and np.any((g1 > 0) & (g1 < 1))


def test_window_power_and_effective_window():
    w = gaps.segment_window(1000, 0.05)
    assert 0.95 < gaps.window_power(w) <= 1.0
    gate = gaps.gate_from_mask(gaps.periodic_mask(1000, 100, 500), 1.0)
    we = gaps.effective_window(gate, w)
    assert we.shape == (1000,)
    assert np.all(we[gate == 0] == 0.0)
