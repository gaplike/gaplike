"""
Paper configuration + thin adapters over the `gaplike` package.

v5 == v4 physics, refactored: every algorithm (windows, windowed covariances,
likelihood tiers, analytic Fisher/bias/scatter machinery, noise simulation,
lisabeta waveform adapter) now lives in `gaplike`; this module pins the
paper's configuration (12 h TDI2 A/E segment, IMRPhenomHM MBHB, analysis band
[1e-4, 0.031] Hz) and re-exports the exact API used by the pipeline
(`scenarios.py`, `pe.py`, `plots.py`, `scenC.py`, ...).

Conventions (unchanged from v4):
  * data vector y = sqrt(2 dt / M) rfft(w_eff x)[band]  (one-sided-PSD units),
  * noise parameters lam = log10 deviations of the TM / OMS amplitudes,
  * time-domain template h_td = irfft(h~ / dt)  (CFT -> DFT rule),
  * channels A + E: identical covariances, independent noises.
"""
import numpy as np

import gaplike as gl
from diagnostics import safe_inv as _safe_inv                  # noqa: F401
from diagnostics import build_truth as _gl_build_truth
from diagnostics import snr_windowed                            # noqa: F401
import diagnostics as _diag

LN10 = np.log(10.0)
C_LIGHT = gl.psd.C_LIGHT
L_M = 2.5e9
AVG_D = L_M / C_LIGHT

# ============================== configuration ==============================
DT = 15.0                       # [s] cadence (Nyquist 1/30 Hz > 0.031 Hz)
SEG_HOURS = 12.0                # [h] segment length
M = int(round(SEG_HOURS * 3600 / DT))
T_OBS = M * DT
FS = 1.0 / DT
DF = 1.0 / T_OBS
ANALYSIS_ALPHA = 0.05           # segment-edge Tukey taper (always on)

F_LO_BAND = 1.0e-4              # [Hz] analysis band
F_HI_BAND = 3.1e-2              # includes the TDI2 null at c/(8L) ~ 0.02998 Hz

TDI_GEN = 2
NCH = 2                         # analysed channels: A, E

APPROX = "IMRPhenomHM"          # higher modes: (2,2),(2,1),(3,3),(3,2),(4,4),(4,3)
WF_KW = dict(TDI='TDI2AET', TDIrescaled=False, approximant=APPROX)

TM0, OMS0 = 3e-15, 15e-12       # reference ASDs; lam = log10(A/A0)

# grids
freqs = np.fft.rfftfreq(M, DT)
fgrid = np.fft.fftfreq(M, DT)
band = (freqs > F_LO_BAND) & (freqs < F_HI_BAND)
idx_band = np.where(band)[0]
NB = idx_band.size
fb = freqs[band]
scale = np.sqrt(2.0 / (FS * M))             # rfft -> periodogram normalization
th_grid = np.arange(M) * DT / 3600.0        # segment time grid [h]


# ============================== noise model ===============================
_PSD = gl.psd.lisa_tdi2_ae(tm_asd=TM0, oms_asd=OMS0, arm_m=L_M,
                           generation=TDI_GEN)


def psd_components(f, gen=TDI_GEN):
    """(S_TM, S_OMS): the two A/E-channel PSD components at reference amplitudes."""
    assert gen == TDI_GEN
    return _PSD["tm"](f), _PSD["oms"](f)


def SA_instr(f):
    s_tm, s_oms = psd_components(f)
    return s_tm + s_oms


S2_TM = gl.psd.two_sided_grid(_PSD["tm"], M, DT)
S2_OMS = gl.psd.two_sided_grid(_PSD["oms"], M, DT)
S2 = S2_TM + S2_OMS
S_SEG = gl.psd.one_sided_grid(SA_instr, M, DT)
STM_BARE, SOMS_BARE = psd_components(fb)


def gen_noise_td(rng, nch=NCH):
    """(nch, M) independent stationary noise realizations (time domain)."""
    return gl.simulate.noise_td(S_SEG, DT, rng, nch=nch)


# ===================== windows / controlled gaps ===========================
_planck_edge = gl.gaps.planck_edge
W_SEG = gl.gaps.segment_window(M, ANALYSIS_ALPHA)


def controlled_gaps(gaps, taper_h=0.12):
    """Gate with gaps at given (centre [h], duration [h]), Planck-tapered edges."""
    return gl.gaps.gate_from_gaps(gaps, taper_h, t=th_grid)


def make_weff(gaps, taper_h=0.12):
    """Effective window w_eff = w_seg * controlled_gaps(...)."""
    g = controlled_gaps(gaps, taper_h) if gaps else np.ones(M)
    return W_SEG * g, g


# ======================= windowed FD covariance (paper Eq. 40) =============
def freq_cov(we, S2loc, idx):
    """Convolved covariance of the windowed rfft, restricted to bins idx."""
    return gl.covariance.full_covariance(we, S2loc, idx)


def conv_diag(we, S2loc):
    """Exact diagonal of the convolved covariance, O(N log N)."""
    return gl.covariance.convolved_diag(we, S2loc)


# ============================== waveform ====================================
SIG_NAMES = ["Mtot", "q", "chi1", "chi2", "log10dL", "iota", "phi",
             "lambda", "beta", "psi", "tc_frac"]
NSIG = len(SIG_NAMES)
NOISE_NAMES = ["lam_tm", "lam_oms"]

# ---- truth: the source of gaps_mbhb_mle.ipynb, d_L fixed at 40 Gpc ----
THETA_SIG_TRUE = np.array([2.5e7, 1.5, 0.3, 0.1, np.log10(40.0),
                           0.6, 1.0, 1.0, 0.4, 0.3,
                           0.80])

# finite-difference steps; log10dL and tc_frac derivatives are analytic
FD_STEPS = {"Mtot": 30.0, "q": 1e-4, "chi1": 1e-4, "chi2": 2e-4,
            "log10dL": None, "iota": 1e-3, "phi": 1e-3, "lambda": 1e-3,
            "beta": 1e-3, "psi": 1e-3, "tc_frac": None}

_params_dict = gl.waveform.default_mbhb_params
_WF = None


def wf_AE_fullgrid(theta):
    """(2, M//2+1) band-limited TDI2 A,E FD waveforms (numpy convention),
    summed over all IMRPhenomHM modes; t_c as an FD phase (gaplike adapter,
    lisabeta evaluated on the band bins only)."""
    global _WF
    if _WF is None:
        _WF = gl.waveform.lisabeta_mbhb_ae(M, DT, F_LO_BAND, F_HI_BAND,
                                           wf_kw=WF_KW)
    return _WF.fd(theta)


def y_of_td(x_td, weff):
    """Banded, periodogram-normalized rfft of the windowed series."""
    return gl.covariance.transform(x_td, weff, idx_band, DT)


def template_y(theta_sig, weff):
    """(2, NB) windowed banded template (CFT->DFT: h_td = irfft(h~/dt))."""
    ht = wf_AE_fullgrid(theta_sig)
    h_td = np.fft.irfft(ht / DT, n=M, axis=-1)
    return y_of_td(h_td, weff)


def signal_derivs(weff):
    """h0: (2, NB); dh: (2, NB, NSIG). Central finite differences; analytic for
    log10dL (dh = -ln10 h) and tc_frac (pure FD phase: dh~ = -2i pi f T h~)."""
    h0 = template_y(THETA_SIG_TRUE, weff)
    dh = np.zeros((2, NB, NSIG), complex)
    for a, name in enumerate(SIG_NAMES):
        st = FD_STEPS[name]
        if name == "log10dL":
            dh[:, :, a] = -LN10 * h0
            continue
        if name == "tc_frac":
            ht = wf_AE_fullgrid(THETA_SIG_TRUE)
            dht = ht * (-2j * np.pi * freqs * T_OBS)[None, :]
            d_td = np.fft.irfft(dht / DT, n=M, axis=-1)
            dh[:, :, a] = y_of_td(d_td, weff)
            continue
        tp = THETA_SIG_TRUE.copy(); tp[a] += st
        tm = THETA_SIG_TRUE.copy(); tm[a] -= st
        dh[:, :, a] = (template_y(tp, weff) - template_y(tm, weff)) / (2 * st)
    return h0, dh


# ===================== likelihoods / analytics (gaplike) =====================
class FullCovTrick(gl.likelihood.FullCovariance):
    """v4-compatible facade of gaplike.likelihood.FullCovariance
    (constructor (CT, CO); loglike(a_tm, a_oms, resid))."""

    def __init__(self, CT, CO, rtol_rank=1e-8):
        super().__init__([CT, CO], rtol_rank=rtol_rank)

    @property
    def logdet_COr(self):
        return self.logdet_C1

    def loglike(self, a_tm, a_oms, resid):
        return super().loglike(resid, (a_tm, a_oms))


class ApproxModel(_diag.ApproxModel):
    """v4-compatible facade (default nch = NCH)."""

    def __init__(self, comps, dh, truth, rcond=1e-13, nch=NCH):
        super().__init__(comps, dh, truth, rcond=rcond, nch=nch)


def build_truth(true_components, dh, rcond=1e-13, nch=NCH):
    return _gl_build_truth(true_components, dh, rcond=rcond, nch=nch)


# ============================== SNR utilities ===============================
def snr_whittle(ht):
    m = band.copy()
    return float(np.sqrt(sum(4 * DF * np.sum(np.abs(ht[c][m])**2 / S_SEG[m])
                             for c in range(ht.shape[0]))))
