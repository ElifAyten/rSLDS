import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# plot discrete states 1
def plot_discrete_states(
    z_states, time_vec, shock_times=None, *,
    palette="Set1", lw=4, min_duration=0.5,
    dot_style="line", markersize=8,
    title=None, figsize=(8, 2)
):
    z_states = np.asarray(z_states, int)
    time_vec = np.asarray(time_vec, float)
    T, K = len(z_states), z_states.max() + 1
    colors = sns.color_palette(palette, n_colors=K)


    change_idx = np.where(np.diff(z_states) != 0)[0]
    starts = np.concatenate(([0], change_idx + 1))     # inclusive
    ends   = np.concatenate((change_idx + 1, [T]))      # exclusive

    fig, ax = plt.subplots(figsize=figsize)

    for s, e in zip(starts, ends):
        k        = z_states[s]
        t_start  = time_vec[s]
        t_end    = time_vec[e - 1]          # last sample in the run
        duration = t_end - t_start

        if duration < min_duration:
            if dot_style == "dot":
                ax.plot(0.5*(t_start + t_end), k, 'o',
                        color=colors[k], markersize=markersize, zorder=3)
            else:  # mini bar
                ax.hlines(k, t_start, t_start + min_duration,
                          color=colors[k], lw=lw, alpha=.8, zorder=3)
        else:
            ax.hlines(k, t_start, t_end,
                      color=colors[k], lw=lw, alpha=.8)

    # footshock markers
    if shock_times is not None:
        shock_times = np.squeeze(shock_times)
        if shock_times.shape == z_states.shape:
            s_times = time_vec[shock_times.astype(bool)]
        else:
            s_times = shock_times.astype(float)
        for ts in s_times:
            if time_vec[0] <= ts <= time_vec[-1]:
                ax.axvline(ts, color='k', ls='--', lw=1)

    # plotting
    ax.set_ylim(-0.5, K - 0.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("time (s)")
    ax.set_title(title or "Discrete-state runs")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    return fig, ax
