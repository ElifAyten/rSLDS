# plot_discrete_states_with_speed.py  (overwrite)
import numpy as np, matplotlib.pyplot as plt, seaborn as sns

__all__ = ["plot_discrete_states_with_speed"]

def plot_discrete_states_with_speed(
    z_states, time_vec, speed, shock_times=None,
    *, palette="Set1", lw=4, title=None, figsize=(12,3),
    speed_color="black", speed_alpha=.6, speed_scale=.8, speed_offset=.3
):
    """
    Horizontal bars = discrete states, thin black trace = speed (normalised).
    Returns the Matplotlib figure to let callers save() it.
    """
    z_states = np.asarray(z_states, int)
    time_vec = np.asarray(time_vec, float)
    speed    = np.asarray(speed,  float)
    assert z_states.shape == time_vec.shape == speed.shape

    # --- helpers ------------------------------------------------------------
    speed_norm = (speed - speed.min()) / (speed.ptp())    # 0‥1
    K          = z_states.max() + 1
    colors     = sns.color_palette(palette, n_colors=K)

    change     = np.where(np.diff(z_states) != 0)[0]
    starts     = np.concatenate(([0], change - 1)).clip(0)          # inclusive
    ends       = np.concatenate((change, [len(z_states)-1])).clip(0)

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    # state runs
    for s, e in zip(starts, ends):
        k = z_states[s]
        ax.hlines(k, time_vec[s], time_vec[e], color=colors[k], lw=lw, alpha=.8)

    # speed (floats above last state)
    ax.plot(time_vec,
            speed_norm*speed_scale + (K-1) + speed_offset,
            color=speed_color, alpha=speed_alpha, label="speed (norm)")

    # shock markers
    if shock_times is not None:
        shock_times = np.squeeze(shock_times)
        s_times = (time_vec[shock_times.astype(bool)]
                   if shock_times.shape == z_states.shape
                   else np.asarray(shock_times, float))
        for ts in s_times:
            if time_vec[0] <= ts <= time_vec[-1]:
                ax.axvline(ts, color='k', ls='--', lw=1)

    ax.set_ylim(-.5, K+.8)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("time (s)")
    ax.set_title(title or "discrete states + speed")
    ax.legend()
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig, ax       

