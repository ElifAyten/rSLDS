import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.figure import Figure          # <- NEW

__all__ = ["plot_latents_speed_shocks"]

def plot_latents_speed_shocks(
    x_latents,              # (T, D)
    speed,                  # (T,)
    footshock_input,        # shape (T,) OR list/array of times
    duration,               # seconds
    *,
    smooth_sigma_s  = 0.0,  # Gaussian σ **in seconds**
    integrate_latent = True,
    latent_color     = "black",
    speed_color      = "tab:blue",
    figsize          = (10, 4),
    auto_scale       = True,     # NEW – rescale speed to latent range
    return_figs      = False,    # NEW – return list[Figure] instead of show
):
    """
    Plot every latent together with speed (+ optional shock lines).

    Returns
    -------
    None                – if return_figs=False  (plots are just shown)
    list[matplotlib.figure.Figure]
                        – if return_figs=True   (caller handles save/close)
    """
    T, D      = x_latents.shape
    time_axis = np.linspace(0, duration, T)

    # ------------------------------------------------ decode shock times
    fs = np.squeeze(footshock_input)
    if fs.shape == (T,):
        shock_times = time_axis[fs.astype(bool)]
    else:
        shock_times = np.asarray(fs, dtype=float)

    # ------------------------------------------------ optional smoothing
    if smooth_sigma_s > 0:
        dt = time_axis[1] - time_axis[0]
        sigma_samples = smooth_sigma_s / dt
        x_smooth   = gaussian_filter1d(x_latents, sigma=sigma_samples,
                                       axis=0, mode="reflect")
        speed_smooth = gaussian_filter1d(speed, sigma=sigma_samples,
                                         mode="reflect")
    else:
        x_smooth, speed_smooth = x_latents, speed

    # ------------------------------------------------ plotting loop
    figs: list[Figure] = []
    for d in range(D):
        lat = x_smooth[:, d]
        if integrate_latent:
            lat = np.cumsum(lat) * (duration / T)

        fig, ax_lat = plt.subplots(figsize=figsize)

        # latent trace
        ax_lat.plot(time_axis, lat, color=latent_color, lw=1.5,
                    label=f"latent {d+1}{' (int)' if integrate_latent else ''}")
        ax_lat.set_ylabel("latent value", color=latent_color)
        ax_lat.tick_params(axis="y", labelcolor=latent_color)

        # speed on twin axis
        ax_spd = ax_lat.twinx()
        ax_spd.plot(time_axis, speed_smooth, color=speed_color,
                    lw=1.5, alpha=0.7, label="speed")
        ax_spd.set_ylabel("speed", color=speed_color)
        ax_spd.tick_params(axis="y", labelcolor=speed_color)

        if auto_scale:
            lat_peak  = np.nanpercentile(np.abs(lat), 99)
            spd_peak  = np.nanpercentile(speed_smooth, 99)
            if spd_peak > 0:
                scale = lat_peak / spd_peak
                lo, hi = ax_spd.get_ylim()
                ax_spd.set_ylim(lo * scale, hi * scale)

        # shock markers
        for ts in shock_times:
            ax_lat.axvline(ts, color="red", ls="--", lw=1, alpha=0.5)

        ax_lat.set_xlabel("time (s)")
        ax_lat.set_title(f"Latent {d+1} vs. speed & shocks")

        # merge legends
        h1, l1 = ax_lat.get_legend_handles_labels()
        h2, l2 = ax_spd.get_legend_handles_labels()
        ax_lat.legend(h1 + h2, l1 + l2, loc="upper left")

        plt.tight_layout()

        if return_figs:
            figs.append(fig)
        else:
            plt.show()

    return figs if return_figs else None
