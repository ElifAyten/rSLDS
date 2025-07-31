"""
mutual_information.py 

"""
from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

from scipy.stats import percentileofscore
from sklearn.feature_selection import mutual_info_regression

__all__ = ["latent_signal_mi", "circular_shift", "phase_shuffle"]
__version__ = "0.2.0"

# ------------------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------------------

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    """Roll *arr* by *shift* samples (cyclic)."""
    return np.roll(arr, int(shift) % len(arr), axis=0)


def phase_shuffle(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase‑randomised surrogate with identical power spectrum."""
    fft = np.fft.rfft(sig)
    phase = rng.random(len(fft)) * 2 * np.pi
    return np.fft.irfft(np.abs(fft) * np.exp(1j * phase), n=len(sig))


def _mi(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """Mutual information using scikit‑learn's k‑NN estimator."""
    return mutual_info_regression(
        x.reshape(-1, 1),
        y.astype(np.float64),
        n_neighbors=k,
        random_state=seed,
        discrete_features=False,
    )[0]


def _adaptive_k(n: int) -> int:
    """Choose k for MI estimator (scales with sample size)."""
    return max(1, min(3, n - 1))


def _mi_surrogates(
    x: np.ndarray,
    y: np.ndarray,
    *,
    shuffle: Literal["circular", "strict", "permute"],
    n: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """Return (raw, min_null, thr95, p) MI values."""
    rng = np.random.default_rng(seed)
    k = _adaptive_k(len(x))
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

    return (
        raw,
        null.min(),
        np.percentile(null, 95),
        1.0 - percentileofscore(null, raw) / 100.0,
    )
# --------------------------------------------------------------------------
#  compare MI before vs. after the first foot-shock
# --------------------------------------------------------------------------
def compare_mi_pre_post(
    *,
    latents: np.ndarray,
    signal: np.ndarray,
    time_vec: np.ndarray,
    footshock_mask: np.ndarray | np.ndarray,
    win_s: float = 1.0,
    integrate_latents: bool = True,
    shuffle: Literal["circular", "strict", "permute"] = "circular",
    n_shuffle: int = 500,
    random_state: int = 0,
    plot: bool = True,
):
    """
    Compute mutual information (MI) for each latent dimension **before**
    and **after** the first foot-shock.

    Parameters
    ----------
    latents, signal, time_vec
        Same semantics as `latent_signal_mi`.
    footshock_mask : (T,) bool or 0/1 array
        True/1 on the time-points that belong to foot-shock onsets.
    win_s, integrate_latents, shuffle, n_shuffle, random_state
        Passed straight through to `latent_signal_mi`.
    plot : bool, default True
        If True show a bar-plot comparing pre- and post-shock MI.

    Returns
    -------
    pre_rec, post_rec : structured arrays (see `latent_signal_mi`)
    """
    footshock_mask = np.asarray(footshock_mask, bool)
    if footshock_mask.shape != time_vec.shape:
        raise ValueError("footshock_mask must have the same length as time_vec")

    # ------------------------------------------------------------
    # Find the first shock onset
    # ------------------------------------------------------------
    if not footshock_mask.any():
        raise ValueError("footshock_mask contains no True entries")
    first_shock_idx = np.where(footshock_mask)[0][0]
    t0 = time_vec[first_shock_idx]

    pre_idx  = time_vec < t0
    post_idx = time_vec >= t0

    if pre_idx.sum() < 10 or post_idx.sum() < 10:
        raise ValueError("Too few samples in pre- or post-shock segment")

    pre_rec = latent_signal_mi(
        latents          = latents[pre_idx],
        signal           = signal[pre_idx],
        time_vec         = time_vec[pre_idx],
        win_s            = win_s,
        integrate_latents= integrate_latents,
        shuffle          = shuffle,
        n_shuffle        = n_shuffle,
        random_state     = random_state,
    )

    post_rec = latent_signal_mi(
        latents          = latents[post_idx],
        signal           = signal[post_idx],
        time_vec         = time_vec[post_idx],
        win_s            = win_s,
        integrate_latents= integrate_latents,
        shuffle          = shuffle,
        n_shuffle        = n_shuffle,
        random_state     = random_state + 1,   # different seed
    )

    # ------------------------------------------------------------
    # Optional bar-plot
    # ------------------------------------------------------------
    if plot:
        dims = pre_rec["dim"]
        x    = np.arange(len(dims))
        width = 0.35

        fig, ax = plt.subplots(figsize=(0.8*len(dims)+3, 3))
        ax.bar(x - width/2, pre_rec["MI_raw"], width,
               label="pre-shock",  color="tab:blue", alpha=.8)
        ax.bar(x + width/2, post_rec["MI_raw"], width,
               label="post-shock", color="tab:orange", alpha=.8)

        ax.set_xticks(x); ax.set_xticklabels([f"dim {d}" for d in dims])
        ax.set_ylabel("Mutual information (bits)")
        ax.set_title("Pre- vs. post-shock MI per latent dimension")
        ax.legend();  plt.tight_layout()

    return pre_rec, post_rec


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

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
    """Compute MI between every latent dimension and a 1‑D behavioural signal.

    Parameters
    ----------
    latents : array, shape (T, D)
        Continuous latent trajectories (rows = time points).
    signal : array, shape (T,)
        1‑D behavioural signal sampled at the same times as *latents*.
    time_vec : array, shape (T,)
        Time stamps (seconds) for each sample.
    win_s : float, default 1.0
        Window length in seconds for averaging before MI estimation.
    integrate_latents : bool, default True
        If True integrate (cumulatively sum) each latent prior to windowing.
        This often matches the interpretation of rSLDS velocity latents.
    shuffle : {'circular', 'strict', 'permute'}, default 'circular'
        Null model to build the surrogate distribution.
    n_shuffle : int, default 500
        Number of surrogates to draw.
    random_state : int, default 0
        Seed for the global RNG.

    Returns
    -------
    rec : structured array, shape (D,)
        dtype = [('dim', int), ('MI_raw', float), ('MI_min', float),
                 ('thr95', float), ('p', float)]
    """
    if latents.ndim != 2 or signal.ndim != 1 or time_vec.ndim != 1:
        raise ValueError("latents must be (T, D); signal & time_vec (T,)")
    if not (len(latents) == len(signal) == len(time_vec)):
        raise ValueError("length mismatch between inputs")

    dt = np.diff(time_vec).mean()
    if dt <= 0.0:
        raise ValueError("time_vec must be strictly increasing")

    step = int(round(win_s / dt))
    if step < 1:
        raise ValueError("win_s is shorter than the sampling interval")

    starts = np.arange(0, len(time_vec) - step, step)
    if starts.size < 2:
        raise ValueError("Too few windows – increase recording length or reduce win_s")

    sig_win = np.array([signal[i : i + step].mean() for i in starts])
    rng_root = np.random.default_rng(random_state)

    rec = []
    for d in range(latents.shape[1]):
        trace = latents[:, d]
        if integrate_latents:
            trace = np.cumsum(trace) * dt
        lat_win = np.array([trace[i : i + step].mean() for i in starts])

        raw, mi_min, thr, p = _mi_surrogates(
            lat_win,
            sig_win,
            shuffle=shuffle,
            n=n_shuffle,
            seed=rng_root.integers(2**32 - 1),
        )
        rec.append((d, raw, mi_min, thr, p))

    return np.asarray(
        rec,
        dtype=[
            ("dim", int),
            ("MI_raw", float),
            ("MI_min", float),
            ("thr95", float),
            ("p", float),
        ],
    )
