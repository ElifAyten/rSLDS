"""mutual_information.py – Mutual Information utilities
Compute MI between rSLDS latents and any 1‑D behavioural signal.
Supports three null models: circular shift, phase randomisation (strict),
**and simple random permutation**.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Tuple
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import percentileofscore

__all__ = ["latent_signal_mi", "circular_shift", "phase_shuffle"]
__version__ = "0.1.2"

# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll *arr* by *shift* samples (cyclic)."""
    return np.roll(arr, shift, axis=0)


def phase_shuffle(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Surrogate with identical power spectrum and random phase."""
    fft = np.fft.rfft(sig)
    phase = rng.random(len(fft)) * 2 * np.pi
    return np.fft.irfft(np.abs(fft) * np.exp(1j * phase), n=len(sig))


def _mi(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    return mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=k, random_state=seed)[0]


def _adaptive_k(n: int) -> int:
    return max(1, min(3, n - 1))


def _mi_surrogates(
    x: np.ndarray,
    y: np.ndarray,
    *,
    shuffle: str,
    n: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Compute raw MI and 95‑% surrogate threshold using chosen null model."""
    rng = np.random.default_rng(seed)
    k   = _adaptive_k(len(x))
    raw = _mi(x, y, k, seed)

    null = np.empty(n)
    for i in range(n):
        if shuffle == "circular":
            y_surr = circular_shift(y, rng.integers(len(y)))
        elif shuffle == "strict":
            y_surr = phase_shuffle(y, rng)
        elif shuffle == "permute":
            y_surr = rng.permutation(y)
        else:
            raise ValueError("shuffle must be 'circular', 'strict', or 'permute'")
        null[i] = _mi(x, y_surr, k, seed)

    return raw, np.percentile(null, 95), 1 - percentileofscore(null, raw) / 100


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------

def latent_signal_mi(
    *,
    latents: np.ndarray,
    signal: np.ndarray,
    time_vec: np.ndarray,
    win_s: float = 1.0,
    integrate_latents: bool = True,
    shuffle: Literal["circular", "strict", "permute"] = "circular",
    n_shuffle: int = 500,
    random_state: int = 0,
) -> np.ndarray:
    """Return structured array of MI results for every latent dimension.

    Parameters
    ----------
    latents : (T, D)
    signal  : (T,)
    time_vec: (T,)
    win_s   : window length in seconds
    integrate_latents : cumulative integrate each latent before windowing
    shuffle : null model → 'circular', 'strict' (phase), or 'permute'
    n_shuffle : number of surrogates
    random_state : RNG seed
    """
    if latents.ndim != 2 or signal.ndim != 1 or time_vec.ndim != 1:
        raise ValueError("latents must be (T,D); signal & time_vec (T,)")
    if not (len(latents) == len(signal) == len(time_vec)):
        raise ValueError("length mismatch")

    dt   = np.diff(time_vec).mean()
    step = int(round(win_s / dt))
    if step < 1:
        raise ValueError("win_s < sampling interval")

    starts  = np.arange(0, len(time_vec) - step, step)
    sig_win = np.array([signal[i:i+step].mean() for i in starts])
    rng_root = np.random.default_rng(random_state)

    rec = []
    for d in range(latents.shape[1]):
        trace = latents[:, d]
        if integrate_latents:
            trace = np.cumsum(trace) * dt
        lat_win = np.array([trace[i:i+step].mean() for i in starts])
        raw, thr, p = _mi_surrogates(
            lat_win, sig_win,
            shuffle=shuffle,
            n=n_shuffle,
            seed=rng_root.integers(2**32 - 1),
        )
        rec.append((d, raw, thr, p))

    return np.asarray(rec, dtype=[("dim", int), ("MI_raw", float), ("thr95", float), ("p", float)])
