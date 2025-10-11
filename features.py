import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq
from numpy.polynomial import Polynomial

def extract_stat_features(signal):
    """Basic statistical descriptors of a 1D signal."""
    try:
        np.sqrt(np.mean(signal**2))
    except RuntimeWarning:
        print(signal)
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "skew": skew(signal),
        "kurtosis": kurtosis(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "peak": np.max(signal),
        "min": np.min(signal),
        "range": np.max(signal) - np.min(signal),
    }

def extract_fft_features(signal, fs, n_bands=5):
    """FFT band energies + dominant frequency."""
    N = len(signal)
    freqs = rfftfreq(N, 1/fs)
    fft_vals = np.abs(rfft(signal))**2

    # Split spectrum into bands
    band_edges = np.linspace(0, freqs[-1], n_bands+1)
    band_powers = []
    for i in range(n_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
        band_powers.append(np.sum(fft_vals[mask]))

    # Dominant frequency
    dom_freq = freqs[np.argmax(fft_vals)]
    
    feats = {f"fft_band_{i}": band_powers[i] for i in range(n_bands)}
    feats["dom_freq"] = dom_freq
    return feats

def extract_poly_features(x, y, degree=2):
    """Fit polynomial (e.g., P = aV^2 + bV + c) and return coefficients."""
    try:
        coeffs = Polynomial.fit(x, y, degree).convert().coef
    except Exception:
        coeffs = np.zeros(degree+1)
    return {f"poly_coef_{i}": coeffs[i] for i in range(len(coeffs))}

def extract_features(window, fs, voltage=None, power=None):
    """
    Extract engineered features from a time window.
    window : np.ndarray, shape [channels, timesteps]
    fs     : sampling frequency
    voltage, power: optional signals for domain-specific poly fit
    """
    feats = {}

    # Loop over each channel (e.g., voltage, current, freq)
    for ch in range(window.shape[0]):
        sig = window[ch, :]
        feats.update({f"ch{ch}_{k}": v for k, v in extract_stat_features(sig).items()})
        # feats.update({f"ch{ch}_{k}": v for k, v in extract_fft_features(sig, fs).items()})

    # # Domain-specific: fit polynomial P(V) = aV^2 + bV + c
    # if voltage is not None and power is not None:
    #     feats.update(extract_poly_features(voltage, power, degree=2))

    return feats
