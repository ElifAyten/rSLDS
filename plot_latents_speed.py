
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__all__ = ["plot_latents_speed_shocks"]

def plot_latents_speed_shocks(
    x_latents,              # (T, D)  latent matrix
    speed,                  # (T,)    running-speed trace
    footshock_input,        # (T,) binary ❶ or (N,) shock-timestamps ❷
    duration,               # total time span in seconds
    *,
    smooth_sigma_s = 0.0,   # Gaussian σ in **seconds** (0 = no smoothing)
    integrate_latent = True,
    latent_color = "black",
    speed_color  = "tab:blue",
    figsize = (10,4)
):
    """
    Plot each latent (optionally integrated) together with speed and shock lines.

    ❶ If `footshock_input.shape == (T,)` → treat as 0/1 vector in sample space
    ❷ If it’s shorter → treat as list/array of absolute timestamps (s)
    """
    T, D      = x_latents.shape
    time_axis = np.linspace(0, duration, T)

    # decode shock times 
    fs = np.squeeze(footshock_input)
    if fs.shape == (T,):                      # binary vector case
        shock_times = time_axis[fs.astype(bool)]
    else:                                     # already timestamps
        shock_times = np.asarray(fs, dtype=float)

    # smoothing
    if smooth_sigma_s > 0:
        dt = time_axis[1] - time_axis[0]      # sample interval in s
        sigma_samples = smooth_sigma_s / dt
        x_smooth = gaussian_filter1d(x_latents, sigma=sigma_samples,
                                     axis=0, mode="reflect")
        speed_smooth = gaussian_filter1d(speed, sigma=sigma_samples,
                                         mode="reflect")
    else:
        x_smooth = x_latents
        speed_smooth = speed

    # iterate over latent dims 
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
        ax_spd.plot(time_axis, speed_smooth, color=speed_color, lw=1.5, alpha=0.7,
                    label="speed")
        ax_spd.set_ylabel("speed", color=speed_color)
        ax_spd.tick_params(axis="y", labelcolor=speed_color)

        # foot-shock markers
        for ts in shock_times:
            ax_lat.axvline(ts, color="red", ls="--", lw=1, alpha=0.5)

        ax_lat.set_xlabel("time (s)")
        ax_lat.set_title(f"Latent {d+1} vs. speed & shocks")

        # merged legend
        handles1, labels1 = ax_lat.get_legend_handles_labels()
        handles2, labels2 = ax_spd.get_legend_handles_labels()
        ax_lat.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

        plt.tight_layout()
        plt.show()
