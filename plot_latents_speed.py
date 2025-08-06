# ──────────────────────────────────────────────────────────────
#   helper: latent   + speed   + foot-shock markers
# ──────────────────────────────────────────────────────────────
import numpy as np, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__all__ = ["plot_latents_speed_shocks"]

def plot_latents_speed_shocks(
    *,
    x_latents,                # (T, D)
    speed,                    # (T,)
    footshock_input,          # (T,) binary   or  (N,) timestamps
    duration,                 # total seconds
    integrate_latent=True,
    smooth_sigma_lat_s=0.0,   # latents  (sec σ)  0=none
    speed_smooth_sigma_s=0.0, # speed    (sec σ)  0=none
    latent_color="black",
    speed_color="tab:blue",
    figsize=(10, 4),
    return_figs=False,
):
    """
    One figure per latent dimension.

    speed is plotted on a *twin y-axis* without forcibly matching the
    latent’s amplitude.  Shock onsets ≈ red dashed lines.
    """
    T, D = x_latents.shape
    t_ax = np.linspace(0, duration, T)

    # decode shock times -------------------------------------------------
    fs = np.squeeze(footshock_input)
    if fs.shape == (T,):                        # binary (sample space)
        shock_times = t_ax[fs.astype(bool)]
    else:                                       # already timestamps
        shock_times = np.asarray(fs, float)

    # smoothing ----------------------------------------------------------
    def _smooth(arr, sigma_s):
        if sigma_s <= 0:  return arr
        dt = t_ax[1] - t_ax[0]
        return gaussian_filter1d(arr, sigma=sigma_s/dt, axis=0, mode="reflect")

    x_smooth  = _smooth(x_latents,  smooth_sigma_lat_s)
    sp_smooth = _smooth(speed,      speed_smooth_sigma_s)

    figs = []
    for d in range(D):
        trace = x_smooth[:, d]
        if integrate_latent:
            trace = np.cumsum(trace) * (duration / T)

        fig, ax_lat = plt.subplots(figsize=figsize)

        # latent ---------------------------------------------------------
        ax_lat.plot(t_ax, trace, color=latent_color, lw=1.2,
                    label=f"latent {d+1}{' (int)' if integrate_latent else ''}")
        ax_lat.set_xlabel("time (s)")
        ax_lat.set_ylabel("latent value", color=latent_color)
        ax_lat.tick_params(axis="y", labelcolor=latent_color)

        # speed on twin axis --------------------------------------------
        ax_spd = ax_lat.twinx()
        ax_spd.plot(t_ax, sp_smooth, color=speed_color, lw=1.2, alpha=0.7,
                    label="speed")
        ax_spd.set_ylabel("speed", color=speed_color)
        ax_spd.tick_params(axis="y", labelcolor=speed_color)

        # shocks --------------------------------------------------------
        for ts in shock_times:
            ax_lat.axvline(ts, color="red", ls="--", lw=0.8, alpha=0.5)

        # title & legend -------------------------------------------------
        ax_lat.set_title(f"Latent {d+1} vs. speed & shocks")
        h1,l1 = ax_lat.get_legend_handles_labels()
        h2,l2 = ax_spd.get_legend_handles_labels()
        ax_lat.legend(h1+h2, l1+l2, loc="upper left", frameon=False)

        ax_lat.grid(alpha=.25, linestyle=":")
        plt.tight_layout()

        figs.append(fig)

    return figs if return_figs else None

