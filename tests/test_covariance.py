"""Small-N brute-force checks of the covariance builders."""
import numpy as np

from gaplike import covariance as cov
from gaplike import gaps


def _smooth_S2(n, dt):
    f = np.fft.fftfreq(n, dt)
    S2 = np.zeros(n)
    nz = f != 0
    S2[nz] = 1.0 / (1.0 + (np.abs(f[nz]) / 0.05)**2) + 0.1
    return S2


def _window(n):
    w = gaps.segment_window(n, 0.2)
    w[n // 3:n // 3 + n // 8] = 0.0
    return w


def test_full_covariance_vs_direct():
    """The frequency-domain builder equals the brute-force covariance of the
    exact linear map: y = scale * R diag(w) x with circulant time-domain
    covariance Gamma_ts = gamma((t-s) mod n), so
    E[y y^H] = (2 dt / n) * A Gamma A^H,  A = R diag(w)."""
    n, dt = 64, 2.0
    S2 = _smooth_S2(n, dt)
    w = _window(n)
    idx = np.arange(3, 20)

    gamma = np.real(np.fft.ifft(S2)) / (2.0 * dt)
    t = np.arange(n)
    Gamma = gamma[(t[:, None] - t[None, :]) % n]
    R = np.exp(-2j * np.pi * np.outer(idx, t) / n)
    A = R @ np.diag(w)
    Q_direct = (2.0 * dt / n) * (A @ Gamma @ A.conj().T)

    Q = cov.full_covariance(w, S2, idx)
    assert np.allclose(Q, Q_direct, rtol=1e-9)


def test_convolved_diag_is_diagonal_of_full():
    n, dt = 128, 1.0
    S2 = _smooth_S2(n, dt)
    w = _window(n)
    idx = np.arange(n // 2 + 1)
    Q = cov.full_covariance(w, S2, idx)
    d = cov.convolved_diag(w, S2)
    assert np.allclose(np.real(np.diag(Q)), d, rtol=1e-10)
    assert np.max(np.abs(np.imag(np.diag(Q)))) < 1e-15 * d.max()


def test_no_window_diag_is_psd():
    """With w = 1 the convolved diagonal is the two-sided PSD folded to the
    one-sided grid in E[|y|^2] units: S(f) for interior bins."""
    n, dt = 256, 0.5
    S2 = _smooth_S2(n, dt)
    w = np.ones(n)
    d = cov.convolved_diag(w, S2)
    f = np.fft.rfftfreq(n, dt)
    interior = slice(1, -1)
    assert np.allclose(d[interior], S2[:n // 2 + 1][interior], rtol=1e-10)


def test_restricted_autocov():
    n, dt = 64, 1.0
    S2 = _smooth_S2(n, dt)
    obs = np.where(gaps.periodic_mask(n, 4, 16))[0]
    C = cov.restricted_autocov(S2, obs, dt)
    assert np.allclose(C, C.T)
    gamma = np.real(np.fft.ifft(S2)) / (2 * dt)
    t = np.arange(n)
    Full = gamma[(t[:, None] - t[None, :]) % n]
    assert np.allclose(C, Full[np.ix_(obs, obs)])
    # variance on the diagonal: gamma(0) = sum(S2) df / 2 = mean power
    assert np.allclose(np.diag(C), gamma[0])


def test_transform_shape_and_parseval():
    n, dt = 512, 1.0
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, n))
    w = np.ones(n)
    idx = np.arange(1, n // 2)     # exclude DC and Nyquist
    y = cov.transform(x, w, idx, dt)
    assert y.shape == (2, idx.size)
    # Parseval: sum dt x^2 = sum |y|^2 (+DC/Nyquist terms excluded -> close)
    lhs = np.sum(dt * x**2, axis=-1)
    rhs = np.sum(np.abs(y)**2, axis=-1)
    assert np.allclose(lhs, rhs, rtol=0.05)
