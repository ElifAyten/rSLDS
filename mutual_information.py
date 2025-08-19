"""
mutual_information.py
"""
from __future__ import annotations

import numpy as np
from typing import Literal, Tuple

import matplotlib.pyplot as plt  # used in compare_mi_pre_post
from scipy.stats import percentileofscore
from sklearn.feature_selection import mutual_info_regression

__all__ = ["latent_signal_mi", "circular_shift", "phase_shuffle"]
__version__ = "0.2.1"

# helpers

def circular_shift(arr: np.ndarray, shift: int) -> np.ndarray:
    # rolls the signal in time (wrap-around), preserves shape & autocorrelation, breaks alignment.
    """Roll *arr* by *shift* samples (cyclic)."""
    return np.roll(arr, int(shift) % len(arr), axis=0)


def phase_shuffle(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase‑randomised surrogate with identical power spectrum."""
    fft = np.fft.rfft(sig)
    phase = rng.random(len(fft)) * 2 * np.pi
    return np.fft.irfft(np.abs(fft) * np.exp(1j * phase), n=len(sig))
    """What it does: randomizes the phase of the signal in Fourier space but keeps the power spectrum (energy at each frequency) the same.
Why: makes a surrogate with the same “rhythm/smoothness” but different timing, so any real alignment with x is broken."""


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


# null distribution utility

def _right_tailed_perm_p(raw: float, null: np.ndarray) -> float:
    """Permutation p-value: P(null >= raw) with +1 correction."""
    return (1.0 + np.sum(null >= raw)) / (len(null) + 1.0)
""" permutation p-value: probability that a null MI is ≥ raw MI. Uses a +1 correction to avoid 0."""

def _bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR. Returns (reject, qvals).
    """
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qvals = np.empty_like(q)
    qvals[order] = q
    reject = qvals <= alpha
    return reject, qvals


def _mi_surrogates_full(
    x: np.ndarray,
    y: np.ndarray,
    *,
    shuffle: Literal["circular", "strict", "permute"],
    n: int,
    seed: int,
) -> Tuple[float, np.ndarray, float, float, float, float, float]:
    """
    Compute MI and surrogate distribution.
    Returns: raw, null, null_mean, null_std, thr95, p_perm, z
    """
    rng = np.random.default_rng(seed)
    k = _adaptive_k(len(x))
    raw = _mi(x, y, k, seed)

    null = np.empty(n, dtype=float)
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

    null_mean = null.mean()
    null_std = null.std(ddof=1) if n > 1 else np.nan
    thr95 = np.percentile(null, 95)
    p_perm = _right_tailed_perm_p(raw, null)
    z = (raw - null_mean) / null_std if np.isfinite(null_std) and null_std > 0 else np.nan
    return raw, null, null_mean, null_std, thr95, p_perm, z

# mmi before and after shock
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

    Returns
    -------
    pre_rec, post_rec : structured arrays (see `latent_signal_mi`)
    """
    footshock_mask = np.asarray(footshock_mask, bool)
    if footshock_mask.shape != time_vec.shape:
        raise ValueError("footshock_mask must have the same length as time_vec")

    if not footshock_mask.any():
        raise ValueError("footshock_mask contains no True entries")
    first_shock_idx = np.where(footshock_mask)[0][0]
    t0 = time_vec[first_shock_idx]

    pre_idx = time_vec < t0
    post_idx = time_vec >= t0

    if pre_idx.sum() < 10 or post_idx.sum() < 10:
        raise ValueError("Too few samples in pre- or post-shock segment")

    pre_rec = latent_signal_mi(
        latents=latents[pre_idx],
        signal=signal[pre_idx],
        time_vec=time_vec[pre_idx],
        win_s=win_s,
        integrate_latents=integrate_latents,
        shuffle=shuffle,
        n_shuffle=n_shuffle,
        random_state=random_state,
    )

    post_rec = latent_signal_mi(
        latents=latents[post_idx],
        signal=signal[post_idx],
        time_vec=time_vec[post_idx],
        win_s=win_s,
        integrate_latents=integrate_latents,
        shuffle=shuffle,
        n_shuffle=n_shuffle,
        random_state=random_state + 1,  
    )

    if plot:
        dims = pre_rec["dim"]
        x = np.arange(len(dims))
        width = 0.35

        fig, ax = plt.subplots(figsize=(0.8 * len(dims) + 3, 3))
        ax.bar(x - width / 2, pre_rec["MI_raw"], width, label="pre-shock", alpha=.8)
        ax.bar(x + width / 2, post_rec["MI_raw"], width, label="post-shock", alpha=.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"dim {d}" for d in dims])
        ax.set_ylabel("Mutual information")
        ax.set_title("Pre- vs. post-shock MI per latent dimension")
        ax.legend()
        plt.tight_layout()

    return pre_rec, post_rec



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
    # --- NEW options ---
    return_rich: bool = False,
    fdr_alpha: float | None = 0.05,
) -> np.ndarray:
    """
    Compute MI between every latent dimension and a 1‑D behavioural signal.

    Default (return_rich=False) returns the original fields:
      dtype = [('dim', int), ('MI_raw', float), ('MI_min', float),
               ('thr95', float), ('p', float)]

    If return_rich=True, returns extended fields:
      dtype = [
        ('dim', int),
        ('MI_raw', float),
        ('MI_null_mean', float),
        ('MI_null_std', float),
        ('MI_delta', float),        # MI_raw - mean(null)
        ('thr95', float),           # 95th percentile of null
        ('z', float),               # (raw - mean)/std
        ('p_perm', float),          # permutation p-value
        ('significant_95', bool),   # raw > 95th percentile
        ('q_bh', float),            # BH-adjusted p (NaN if fdr_alpha is None)
        ('reject_bh', bool),        # BH decision at fdr_alpha
      ]
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

    # windowing
    sig_win = np.array([signal[i: i + step].mean() for i in starts])
    rng_root = np.random.default_rng(random_state)

    # accumulate
    rich_rows = []
    simple_rows = []
    pvals = []

    for d in range(latents.shape[1]):
        trace = latents[:, d]
        if integrate_latents:
            trace = np.cumsum(trace) * dt
        lat_win = np.array([trace[i: i + step].mean() for i in starts])

        raw, null, mu, sd, thr95, p_perm, z = _mi_surrogates_full(
            lat_win,
            sig_win,
            shuffle=shuffle,
            n=n_shuffle,
            seed=rng_root.integers(2**32 - 1),
        )

        # original outputs (unchanged for backward compatibility)
        p_pct = 1.0 - percentileofscore(null, raw) / 100.0
        simple_rows.append((d, raw, null.min(), thr95, p_pct))

        # rich outputs
        delta = raw - mu
        sig95 = raw > thr95
        rich_rows.append((d, raw, mu, sd, delta, thr95, z, p_perm, sig95, np.nan, False))
        pvals.append(p_perm)

    if not return_rich:
        return np.asarray(
            simple_rows,
            dtype=[
                ("dim", int),
                ("MI_raw", float),
                ("MI_min", float),
                ("thr95", float),
                ("p", float),
            ],
        )

    # optional BH–FDR across dims
    if fdr_alpha is not None and len(rich_rows) > 1:
        reject, qvals = _bh_fdr(np.array(pvals), alpha=fdr_alpha)
        rich_rows = [
            (dim, MI_raw, MI_mu, MI_sd, MI_delta, thr95, z, p_perm, sig95, q, rej)
            for ((dim, MI_raw, MI_mu, MI_sd, MI_delta, thr95, z, p_perm, sig95, _, _), q, rej)
            in zip(rich_rows, qvals, reject)
        ]
    """ keep the expected fraction of false positives among the ones we call significant ≤ alpha."""

    return np.asarray(
        rich_rows,
        dtype=[
            ("dim", int),
            ("MI_raw", float),
            ("MI_null_mean", float),
            ("MI_null_std", float),
            ("MI_delta", float),
            ("thr95", float),
            ("z", float),
            ("p_perm", float),
            ("significant_95", bool),
            ("q_bh", float),
            ("reject_bh", bool),
        ],
    )


#eturns: raw MI, the null vector, and those summary stats.


""" "circular": keep autocorrelation; just misalign so good when stationarity is OK and you want simple, fast nulls.

"strict" (phase): keep full spectrum; timing scrambled so robust for smooth/oscillatory signals.

"permute": destroy all temporal structure so most conservative for time-dependence (but may be too harsh for strongly autocorrelated data)."""
