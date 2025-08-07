# plot_discrete_states_with_speed.py
# ----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_discrete_states_with_speed"]


def plot_discrete_states_with_speed(
    z_states,           # (T,) ints
    time_vec,           # (T,) seconds
    speed,              # (T,) float
    shock_times=None,   # None | 0/1 vector | timestamps
    *,
    palette="Set1",
    lw=4,
    title=None,
    figsize=(12, 3),
    # ---- speed overlay tuning ------------------------------------------
    show_speed="offset",        # "offset"  (legacy)  or  "twin"
    speed_color="black",
    speed_alpha=0.8,
    speed_linewidth=1.4,
    # …only for “offset” mode
    speed_scale=0.9,
    speed_offset=0.35,
):
    """
    Horizontal bars = contiguous discrete states.
    Speed is over-laid either (a) *offset* above the top state
    (legacy look) or (b) on a right-hand y-axis ("twin").

    Returns
    -------
    fig, ax    – the main Matplotlib objects
    """

    # --------------------- sanity checks ---------------------------------
    z_states = np.asarray(z_states, dtype=int)
    time_vec = np.asarray(time_vec, dtype=float)
    speed    = np.asarray(speed,    dtype=float)

    assert z_states.shape == time_vec.shape == speed.shape, \
        "time_vec, z_states, speed must have identical length"

    # --------------------- basic helpers ---------------------------------
    rng = np.ptp(speed)
    rng = rng if rng > 0 else 1           # guard against constant speed
    speed_norm = (speed - speed.min()) / rng   # 0‥1

    K       = z_states.max() + 1
    colours = sns.color_palette(palette, n_colors=K)

    # contiguous runs -----------------------------------------------------
    change   = np.where(np.diff(z_states) != 0)[0]
    starts   = np.concatenate(([0], change))
    ends     = np.concatenate((change, [len(z_states) - 1]))

    # --------------------------- figure ----------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # state bars
    for s, e in zip(starts, ends):
        k = z_states[s]
        ax.hlines(k, time_vec[s], time_vec[e],
                  color=colours[k], lw=lw, alpha=.85)

    # --------------- speed overlay (two styles) --------------------------
    if show_speed == "offset":
        y_speed = speed_norm * speed_scale + (K - 1) + speed_offset
        ax.plot(time_vec, y_speed,
                color=speed_color,
                alpha=speed_alpha,
                lw=speed_linewidth,
                label="speed (norm.)")

    elif show_speed == "twin":
        ax2 = ax.twinx()
        ax2.plot(time_vec, speed,
                 color=speed_color,
                 alpha=speed_alpha,
                 lw=speed_linewidth,
                 label="speed")
        ax2.set_ylabel("speed", color=speed_color)
        ax2.tick_params(axis="y", labelcolor=speed_color)
    else:
        raise ValueError("show_speed must be 'offset' or 'twin'")

    # foot-shock markers --------------------------------------------------
    if shock_times is not None:
        shock_times = np.squeeze(shock_times)
        s_times = (time_vec[shock_times.astype(bool)]
                   if shock_times.shape == z_states.shape
                   else np.asarray(shock_times, float))
        for ts in s_times:
            if time_vec[0] <= ts <= time_vec[-1]:
                ax.axvline(ts, color="k", ls="--", lw=1)

    # cosmetics -----------------------------------------------------------
    ax.set_ylim(-0.5, K + (1.2 if show_speed == "offset" else 0.5))
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("time (s)")
    ax.set_title(title or "Discrete states + speed")

    if show_speed == "offset":
        ax.legend()
    else:  # twin axis: merge legends nicely
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig, ax

