import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

__all__ = ["plot_latents_pupil_shocks"]

def plot_latents_pupil_shocks(
    x_latents,              # (T, D)
    pupil_diameter,         # (T,)
    footshock_input,        # (T,) 0/1  OR  (N,) timestamps
    duration,               # seconds
    *,
    smooth_sigma_s = 0.0,   # Gaussian σ (seconds)  0 = no smoothing
    integrate_latent = True,
    latent_color = "black",
    pupil_color  = "tab:purple",
    figsize = (10,4)
):
    """
    Plot each latent dimension (optionally integrated) together with pupil
    diameter and foot-shock markers, in the same style as the speed plot.
    """
    T, D      = x_latents.shape
    time_axis = np.linspace(0, duration, T)

    # decode shock times 
    fs = np.squeeze(footshock_input)
    if fs.shape == (T,):                     # binary vector case
        shock_times = time_axis[fs.astype(bool)]
    else:                                    # already timestamps
        shock_times = np.asarray(fs, dtype=float)

    # smoothing 
    if smooth_sigma_s > 0:
        dt = time_axis[1] - time_axis[0]     # sample interval (s)
        sigma_samples = smooth_sigma_s / dt
        x_smooth  = gaussian_filter1d(x_latents,     sigma=sigma_samples,
                                      axis=0, mode="reflect")
        pup_smooth= gaussian_filter1d(pupil_diameter,sigma=sigma_samples,
                                      mode="reflect")
    else:
        x_smooth  = x_latents
        pup_smooth= pupil_diameter

    # iterate over latent dimensions 
    for d in range(D):
        lat = x_smooth[:, d]
        if integrate_latent:
            lat = np.cumsum(lat) * (duration / T)   # integrate in “units·s”

        fig, ax_lat = plt.subplots(figsize=figsize)

        # latent trace (left y-axis)
        ax_lat.plot(time_axis, lat, color=latent_color, lw=1.5,
                    label=f"latent {d+1}{' (int)' if integrate_latent else ''}")
        ax_lat.set_ylabel("latent value", color=latent_color)
        ax_lat.tick_params(axis="y", labelcolor=latent_color)

        # pupil trace (right y-axis)
        ax_pup = ax_lat.twinx()
        ax_pup.plot(time_axis, pup_smooth, color=pupil_color, lw=1.3, alpha=0.8,
                    label="pupil diameter")
        ax_pup.set_ylabel("pupil (a.u.)", color=pupil_color)
        ax_pup.tick_params(axis="y", labelcolor=pupil_color)

        # foot-shock markers
        for ts in shock_times:
            ax_lat.axvline(ts, color="red", ls="--", lw=1, alpha=0.5)

        ax_lat.set_xlabel("time (s)")
        ax_lat.set_title(f"Latent {d+1} vs. pupil & shocks")

        # combined legend
        h1,l1 = ax_lat.get_legend_handles_labels()
        h2,l2 = ax_pup.get_legend_handles_labels()
        ax_lat.legend(h1+h2, l1+l2, loc="upper left")

        plt.tight_layout(); plt.show()
