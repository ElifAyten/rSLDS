# ─────────────────────────────────────────────────────────────────────────────
#  helper_mi.py   •  drop this in the same folder as mutual_information.py
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mutual_information as mi               # ← the module you showed

# make sure the helper “knows” about matplotlib only once
if not hasattr(mi, "plt"):
    mi.plt = plt


def mi_pre_post_plot(
    *,
    latents: np.ndarray,
    signal: np.ndarray,
    time_vec: np.ndarray,
    footshock_mask: np.ndarray,
    win_s: float          = 0.3,
    integrate_latents: bool = False,
    shuffle: str          = "circular",
    n_shuffle: int        = 500,
    random_state: int     = 0,
    color_pre: str        = "#3B5BA9",
    color_post: str       = "#C43B3B",
    figsize               = None,
    csv_path: str | None  = None,
):
    """
    Run MI analysis pre- vs post-shock and return a (table, fig) tuple.

    Parameters
    ----------
    latents, signal, time_vec
        Same definitions as in `mutual_information.latent_signal_mi`.
    footshock_mask
        Boolean 0/1 array, `True` at shock onsets.
    win_s, integrate_latents, shuffle, n_shuffle, random_state
        Passed straight through to `compare_mi_pre_post`.
    color_pre, color_post
        Matplotlib colours for the paired bars.
    figsize
        Figure size, default adapts to number of dims.
    csv_path
        If provided, save the result table as CSV.

    Returns
    -------
    table : pandas.DataFrame
    fig   : matplotlib.figure.Figure
    """
    # ── clean NaNs in the behavioural signal ──────────────────────────────
    sig = np.nan_to_num(signal, nan=np.nanmean(signal))

    # ── run analysis (no built-in plotting) ───────────────────────────────
    pre, post = mi.compare_mi_pre_post(
        latents           = latents,
        signal            = sig,
        time_vec          = time_vec,
        footshock_mask    = footshock_mask,
        win_s             = win_s,
        integrate_latents = integrate_latents,
        shuffle           = shuffle,
        n_shuffle         = n_shuffle,
        random_state      = random_state,
        plot              = False,
    )

    table = pd.DataFrame({
        "dim"     : pre["dim"],
        "MI_pre"  : pre["MI_raw"],
        "MI_post" : post["MI_raw"],
        "thr95"   : post["thr95"],
        "p_post"  : post["p"],
        "ΔMI"     : post["MI_raw"] - pre["MI_raw"],
    })

    if csv_path:
        table.to_csv(csv_path, index=False)

    # ── paired-bar figure ─────────────────────────────────────────────────
    dims  = table["dim"]
    x     = np.arange(len(dims))
    w     = 0.35
    if figsize is None:
        figsize = (0.85*len(dims)+3, 3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x-w/2, table["MI_pre"],  w, color=color_pre,  label="pre-shock")
    ax.bar(x+w/2, table["MI_post"], w, color=color_post, label="post-shock")

    # stars where post-shock MI > null threshold
    for xi, raw, thr in zip(x, table["MI_post"], table["thr95"]):
        if raw > thr:
            ax.text(xi+w/2, raw, "★", ha="center", va="bottom",
                    color=color_post, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([f"dim {d}" for d in dims])
    ax.set_ylabel("Mutual information (bits)")
    ax.set_title(f"Pre- vs post-shock MI ({int(win_s*1e3)} ms windows, "
                 f'{"cum" if integrate_latents else "raw"} latents)')
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return table, fig
