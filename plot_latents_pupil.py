import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__all__ = ["plot_latents_pupil_shocks"]

def _decode_shock_times_pupil(footshock_input, t_ax):
    fs = np.squeeze(footshock_input)
    if fs.shape == t_ax.shape:          # sample-aligned 0/1
        return t_ax[fs.astype(bool)]
    return np.asarray(fs, dtype=float).ravel()  # timestamps

def _smooth1d_pupil(arr, sigma_s, dt, axis=0):
    if sigma_s is None or sigma_s <= 0:
        return arr
    return gaussian_filter1d(arr, sigma=sigma_s/dt, axis=axis, mode="reflect")


def plot_latents_pupil_shocks(
    *,
    x_latents,                 # (T, D)
    pupil_diameter,            # (T,)
    footshock_input,           # (T,) binary OR (N,) timestamps
    duration,                  # total seconds
    integrate_latent=True,
    smooth_sigma_lat_s=0.0,    # seconds; 0 = none (latents)
    pupil_smooth_sigma_s=0.0,  # seconds; 0 = none (pupil)
    # backwards-compat alias (optional):
    smooth_sigma_s=None,       # if provided, overrides pupil_smooth_sigma_s
    latent_color="black",
    pupil_color="tab:purple",
    figsize=(10, 4),
    return_figs=False,
):
    """
    One figure per latent dimension with pupil on a twin axis and shock markers.
    Returns list[Figure] if return_figs=True, else None.
    """
    # allow legacy arg name
    if smooth_sigma_s is not None:
        pupil_smooth_sigma_s = smooth_sigma_s

    x_latents      = np.asarray(x_latents)
    pupil_diameter = np.asarray(pupil_diameter)
    T, D = x_latents.shape
    if pupil_diameter.shape != (T,):
        raise ValueError("pupil_diameter must have shape (T,) to match x_latents")

    # time base
    t_ax = np.linspace(0, duration, T, endpoint=False)
    dt = duration / T

    shock_times = _decode_shock_times_pupil(footshock_input, t_ax)

    # smoothing
    x_smooth   = _smooth1d_pupil(x_latents,      smooth_sigma_lat_s,   dt, axis=0)
    pup_smooth = _smooth1d_pupil(pupil_diameter, pupil_smooth_sigma_s, dt, axis=0)

    figs = []
    for d in range(D):
        trace = x_smooth[:, d]
        if integrate_latent:
            trace = np.cumsum(trace) * dt  # proper integral

        fig, ax_lat = plt.subplots(figsize=figsize)

        # latent (left y)
        ax_lat.plot(t_ax, trace, color=latent_color, lw=1.2,
                    label=f"latent {d+1}{' (integrated)' if integrate_latent else ''}")
        ax_lat.set_xlabel("time (s)")
        ax_lat.set_ylabel("latent value", color=latent_color)
        ax_lat.tick_params(axis="y", labelcolor=latent_color)

        # pupil (right y)
        ax_pup = ax_lat.twinx()
        ax_pup.plot(t_ax, pup_smooth, color=pupil_color, lw=1.2, alpha=0.8, label="pupil diameter")
        ax_pup.set_ylabel("pupil (a.u.)", color=pupil_color)
        ax_pup.tick_params(axis="y", labelcolor=pupil_color)

        # shock markers
        for ts in shock_times:
            ax_lat.axvline(ts, color="red", ls="--", lw=0.8, alpha=0.5)

        # styling
        ax_lat.set_title(f"Latent {d+1} vs. pupil & shocks")
        h1,l1 = ax_lat.get_legend_handles_labels()
        h2,l2 = ax_pup.get_legend_handles_labels()
        ax_lat.legend(h1+h2, l1+l2, loc="upper left", frameon=False)
        ax_lat.grid(alpha=.25, linestyle=":")
        fig.tight_layout()
        figs.append(fig)

    return figs if return_figs else None
