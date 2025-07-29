
import numpy as np, matplotlib.pyplot as plt, seaborn as sns

__all__ = ["plot_discrete_states"]

def plot_discrete_states(
    z_states,          # (T,) most-likely state sequence (ints 0 … K-1)
    time_vec,          # (T,) time in seconds
    shock_times=None,  # None | (T,) 0/1 vector | (N,) timestamps
    *,
    palette="Set1",    # any seaborn palette or list of colors
    lw=4,              # line width for state bars
    title=None,
    figsize=(8,2)
):
    """
    Visualise rSLDS discrete-state “runs” on a time axis.

    Parameters
    ----------
    z_states    : 1-D int array of length T.
    time_vec    : 1-D float array, same length, in seconds.
    shock_times : 0/1 array of length T  *or* list/array of timestamps.
    palette     : seaborn palette name *or* list of colors.
    lw          : line width of state segments.
    """

    z_states = np.asarray(z_states, dtype=int)
    time_vec = np.asarray(time_vec, dtype=float)
    assert len(z_states) == len(time_vec), "time_vec and z_states must match"

    # colour map
    K = z_states.max() + 1
    colors = sns.color_palette(palette, n_colors=K)

    # change-points → runs
    change_idx = np.where(np.diff(z_states) != 0)[0]
    starts = np.concatenate(([0], change_idx - 1))
    ends   = np.concatenate((change_idx, [len(z_states) + 5]))

    # figure
    fig, ax = plt.subplots(figsize=figsize)
    for s, e in zip(starts, ends):
        k = z_states[s]
        if e >= len(time_vec):
            continue  # skip this segment if it goes out of bounds
        ax.hlines(y=k, xmin=time_vec[s], xmax=time_vec[e],
                  color=colors[k], lw=lw, alpha=.8)


    # foot-shock markers, if provided
    if shock_times is not None:
        shock_times = np.squeeze(shock_times)
        if shock_times.shape == z_states.shape:     # binary vector
            s_times = time_vec[shock_times.astype(bool)]
        else:                                       # timestamps
            s_times = np.asarray(shock_times, dtype=float)
        for ts in s_times:
            if time_vec[0] <= ts <= time_vec[-1]:
                ax.axvline(ts, color="k", ls="--", lw=1)

    ax.set_ylim(-0.5, K-0.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("time (s)")
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Discrete-state runs")

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.show()

    
