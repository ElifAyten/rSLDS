# ── src/plot_latents_with_footshock.py ────────────────────────────────
import os, numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__all__ = ["load_and_plot_latents"]

# ---------------------------------------------------------------------
def _plot_latents(latents, shocks, *,
                  time_axis=None,
                  integrate=True,
                  smooth_sigma=0.0,
                  colors=("tab:blue",),
                  figsize=(10,4)):
    """
    Internal plotting helper – works on plain arrays.
    """
    T, D = latents.shape
    if time_axis is None:
        time_axis = np.linspace(0, T-1, T)     # generic x-axis

    shock_times = time_axis[np.squeeze(shocks > 0)]

    for d in range(D):
        fig, ax = plt.subplots(figsize=figsize)
        trace = latents[:, d]

        if smooth_sigma > 0:
            trace = gaussian_filter1d(trace, sigma=smooth_sigma)
        if integrate:
            trace = np.cumsum(trace)

        col = colors[d % len(colors)]
        ax.plot(time_axis, trace, color=col, label=f"latent {d+1}")
        ax.set_xlabel("time (index)" if time_axis is None else "time (s)")
        ax.set_ylabel("value")
        ttl = f"latent {d+1}"
        ttl += " (int)" if integrate else ""
        ax.set_title(ttl)

        # red shock markers
        for st in shock_times:
            ax.axvline(st, color="red", ls="--", lw=1, alpha=0.6)

        ax.legend(); ax.grid(True, alpha=.3)
        plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
def load_and_plot_latents(base_dir,
                          rat_tag,
                          model_sub,
                          *,
                          x_pattern      = "x_hat_{rat}.npy",
                          shock_pattern  = "footshock_input{rat}.npy",
                          integrate      = True,
                          smooth_sigma   = 0.0,
                          colors         = ("tab:blue",),
                          figsize        = (10,4)):
    """
    Find x_hat / footshock .npy files inside
        <base_dir>/<model_sub>/,
    then plot every latent dimension with shock markers.

    Parameters
    ----------
    base_dir  : str
        Root path to search (e.g. '/content/drive/MyDrive').
    rat_tag   : str
        Text that appears in the filenames ('RAT15', 'RAT13', …).
    model_sub : str
        Folder name under base_dir that actually holds the .npy files.
    x_pattern : str
        Filename pattern for latents (default 'x_hat_{rat}.npy').
    shock_pattern : str
        Pattern for shock regressor (default 'footshock_input{rat}.npy').
    integrate : bool
        True  → cumulative sum before plotting.
    smooth_sigma : float
        σ for Gaussian smoothing; 0 disables smoothing.
    """

    model_dir = os.path.join(base_dir, model_sub)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(model_dir)

    x_file   = os.path.join(model_dir, x_pattern.format(rat=rat_tag))
    shock_file = os.path.join(model_dir, shock_pattern.format(rat=rat_tag))

    if not os.path.isfile(x_file):
        raise FileNotFoundError(x_file)
    if not os.path.isfile(shock_file):
        raise FileNotFoundError(shock_file)

    latents = np.load(x_file)            # (T, D)
    shocks  = np.load(shock_file)        # (T,) or (T,1)

    _plot_latents(latents, shocks,
                  integrate=integrate,
                  smooth_sigma=smooth_sigma,
                  colors=colors,
                  figsize=figsize)
