"""EEG preprocessing (Sec. III-F) and db4 wavelet band decomposition (Sec. III-F.1)."""
from fractions import Fraction

import numpy as np
import pywt
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, resample_poly

BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]


def preprocess(eeg, fs, cfg):
    """Band-pass 1-60 Hz (Butterworth, order 4) -> notch -> optional CAR -> resample."""
    x = eeg.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    hi = min(cfg.bandpass[1], 0.45 * fs)
    sos = butter(cfg.filter_order, [cfg.bandpass[0], hi], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    if cfg.notch_hz and cfg.notch_hz < fs / 2:
        b, a = iirnotch(cfg.notch_hz, cfg.notch_q, fs=fs)
        x = filtfilt(b, a, x, axis=1)
    if cfg.common_average_ref:
        x = x - x.mean(axis=0, keepdims=True)
    if fs != cfg.target_fs:
        fr = Fraction(cfg.target_fs / fs).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator, axis=1)
    return x


def wavelet_bands(x, wavelet="db4", level=4):
    """Decompose (channels, samples) into 5 bands, each reconstructed to full length.

    At 128 Hz a 4-level DWT gives A4 (0-4 Hz)=delta, D4 (4-8)=theta, D3 (8-16)=alpha,
    D2 (16-32)=beta, D1 (32-64)=gamma. Reconstruction (rather than raw coefficients) keeps
    every band at the same sampling rate so 4-s windows line up across bands.
    """
    n = x.shape[-1]
    coeffs = pywt.wavedec(x, wavelet, level=level, axis=-1)   # [A4, D4, D3, D2, D1]
    out = {}
    for bi, band in enumerate(BAND_ORDER):
        keep = [c if i == bi else np.zeros_like(c) for i, c in enumerate(coeffs)]
        out[band] = pywt.waverec(keep, wavelet, axis=-1)[..., :n]
    return out
