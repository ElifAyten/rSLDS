import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__all__ = [
    "plot_latents_from_dir",  # convenience one‑liner
    "load_and_plot_latents",  # legacy API
]

# -----------------------------------------------------------------------------
# Internal helper – agnostic to where the data came from
# -----------------------------------------------------------------------------

def _plot_latents(
    latents,
    shocks,
    *,
    time_axis=None,
    integrate=True,
    smooth_sigma=0.0,
    colors=("tab:blue",),
    figsize=(10, 4),
):
    """Plot every latent dimension and overlay foot‑shock onsets (red dashed)."""

    T, D = latents.shape
    if time_axis is None:
        time_axis = np.arange(T)  # generic x‑axis (index)

    shock_times = time_axis[np.squeeze(shocks > 0)]

    for d in range(D):
        fig, ax = plt.subplots(figsize=figsize)
        trace = latents[:, d]

        if smooth_sigma > 0:
            trace = gaussian_filter1d(trace, sigma=smooth_sigma)
        if integrate:
            trace = np.cumsum(trace)

        col = colors[d % len(colors)]
        ax.plot(time_axis, trace, color=col, label=f"Latent {d + 1}")
        ax.set_xlabel("time (index)" if time_axis is None else "time (s)")
        ax.set_ylabel("value")
        ax.set_title(f"Latent {d + 1}")  # removed " (int)" suffix

        for st in shock_times:
            ax.axvline(st, color="red", ls="--", lw=1, alpha=0.6)

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# 1) Convenience entry‑point matching artefacts saved by fit_single_rslds
# -----------------------------------------------------------------------------

def plot_latents_from_dir(
    save_dir,
    *,
    integrate=True,
    smooth_sigma=0.0,
    colors=("tab:blue",),
    figsize=(10, 4),
):
    """Load ``x_hat.npy`` & ``footshock.npy`` from *save_dir* and plot latents."""

    x_file = os.path.join(save_dir, "x_hat.npy")
    shock_file = os.path.join(save_dir, "footshock.npy")

    if not os.path.isfile(x_file):
        raise FileNotFoundError(x_file)
    if not os.path.isfile(shock_file):
        raise FileNotFoundError(shock_file)

    latents = np.load(x_file)
    shocks = np.load(shock_file)

    _plot_latents(
        latents,
        shocks,
        integrate=integrate,
        smooth_sigma=smooth_sigma,
        colors=colors,
        figsize=figsize,
    )


# -----------------------------------------------------------------------------
# 2) Back‑compat wrapper (kept almost identical to the GitHub original)
# -----------------------------------------------------------------------------

def load_and_plot_latents(
    base_dir,
    rat_tag,
    model_sub,
    *,
    x_pattern="x_hat.npy",
    shock_pattern="footshock.npy",
    integrate=True,
    smooth_sigma=0.0,
    colors=("tab:blue",),
    figsize=(10, 4),
):
    """Locate and plot latent trajectories + shock markers."""

    model_dir = os.path.join(base_dir, model_sub)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(model_dir)

    x_file = os.path.join(model_dir, x_pattern.format(rat=rat_tag))
    shock_file = os.path.join(model_dir, shock_pattern.format(rat=rat_tag))

    if not os.path.isfile(x_file):
        raise FileNotFoundError(x_file)
    if not os.path.isfile(shock_file):
        raise FileNotFoundError(shock_file)

    latents = np.load(x_file)
    shocks = np.load(shock_file)

    _plot_latents(
        latents,
        shocks,
        integrate=integrate,
        smooth_sigma=smooth_sigma,
        colors=colors,
        figsize=figsize,
    )

