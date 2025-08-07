# plot_discrete_states_with_speed.py
# ----------------------------------
import numpy as np # Added import here
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_discrete_states_with_speed"]


def plot_discrete_states_with_speed( # Corrected function name to match import
    z_states,           # (T,) discrete labels
    time_vec,           # (T,) seconds
    speed,              # (T,) raw or cleaned speed
    shock_times=None,   # None | 0/1 vector | timestamp list/array
    *,
    palette="Set1",
    lw=4,
    title=None,
    figsize=(12, 3),
    speed_color="black",
    speed_alpha=0.6,
    speed_scale=0.8,
    speed_offset=0.3,
):
    """
    Horizontal bars = contiguous discrete-state runs.
    Thin black (default) trace  = normalised speed.
    Dashed lines                = foot-shock times (if provided).

    Returns
    -------
    (fig, ax) : tuple
        Matplotlib figure & axis so callers can further tweak / save().
    """
    # ---------------------------- basic checks -----------------------------
    z_states = np.asarray(z_states, dtype=int)
    time_vec = np.asarray(time_vec, dtype=float)
    speed    = np.asarray(speed,    dtype=float)

    assert (
        z_states.shape == time_vec.shape == speed.shape
    ), "z_states, time_vec and speed must have identical length"

    # ------------------------ normalise speed -----------------------------
    rng = np.ptp(speed)            # peak-to-peak range (NumPy public API)
    rng = rng if rng > 0 else 1    # avoid division by 0 if speed is constant
    speed_norm = (speed - np.min(speed)) / rng # Use np.min instead of speed.min()

    # ------------------------ colour palette ------------------------------
    K      = z_states.max() + 1
    colours = sns.color_palette(palette, n_colors=K)

    # --------------------- find contiguous state runs ---------------------
    change_idx = np.where(np.diff(z_states) != 0)[0]
    starts = np.concatenate(([0], change_idx + 1))          # inclusive # Corrected index + 1
    ends   = np.concatenate((change_idx, [len(z_states) - 1]))  # inclusive

    # ---------------------------- plotting --------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # draw horizontal bars
    for s, e in zip(starts, ends):
        k = z_states[s]
        ax.hlines(
            y=k,
            xmin=time_vec[s],
            xmax=time_vec[e],
            color=colours[k],
            lw=lw,
            alpha=0.8,
        )

    # overlay normalised speed (above the top state line)
    ax.plot(
        time_vec,
        speed_norm * speed_scale + (K - 1) + speed_offset,
        color=speed_color,
        alpha=speed_alpha,
        label="speed (norm.)",
    )

    # foot-shock markers (optional)
    if shock_times is not None:
        shock_times = np.squeeze(shock_times)
        if shock_times.shape == z_states.shape:                # binary vector
            s_times = time_vec[shock_times.astype(bool)]
        else:                                                  # timestamp list/array
            s_times = np.asarray(shock_times, dtype=float)

        for ts in s_times:
            if time_vec[0] <= ts <= time_vec[-1]:
                ax.axvline(ts, color="k", ls="--", lw=1)

    # --------------------- axis cosmetics ---------------------------------
    ax.set_ylim(-0.5, K + 0.8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("time (s)")
    ax.set_title(title or "Discrete states + speed overlay")
    ax.legend()

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    return fig, ax
