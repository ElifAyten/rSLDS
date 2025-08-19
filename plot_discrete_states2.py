import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plot discrete states 2 
__all__ = ["plot_discrete_states2"]

def plot_discrete_states2(
    z_states,          # (T,) integer labels
    time_vec,          # (T,) seconds
    shock_times=None,  # None | 0/1 vector | timestamp list
    *,
    palette="Set1",
    lw=4,
    title=None,
    figsize=(8, 2),
):
    """Plot contiguous discrete-state runs with optional foot-shock markers."""

    z_states = np.asarray(z_states, dtype=int)
    time_vec = np.asarray(time_vec, dtype=float)
    assert z_states.shape == time_vec.shape, "time_vec and z_states must align"

    K = z_states.max() + 1
    colors = sns.color_palette(palette, n_colors=K)

    
    change_idx = np.where(np.diff(z_states) != 0)[0]
    starts = np.concatenate(([0], change_idx - 1))                 # −1 pad
    ends   = np.concatenate((change_idx, [len(z_states) - 1]))     # clamp

    # make sure no index is negative or beyond last element
    starts = np.clip(starts, 0, len(z_states) - 1)
    ends   = np.clip(ends,   0, len(z_states) - 1)

    # plot runs
    fig, ax = plt.subplots(figsize=figsize)
    for s, e in zip(starts, ends):
        k = z_states[s]
        ax.hlines(k, time_vec[s], time_vec[e],
                  color=colors[k], lw=lw, alpha=.8)

    # fs markres
    if shock_times is not None:
        shock_times = np.squeeze(shock_times)
        if shock_times.shape == z_states.shape:                # binary vector
            s_times = time_vec[shock_times.astype(bool)]
        else:                                                  # timestamps
            s_times = np.asarray(shock_times, dtype=float)
        for ts in s_times:
            if time_vec[0] <= ts <= time_vec[-1]:
                ax.axvline(ts, color="k", ls="--", lw=1)

    ax.set_ylim(-0.5, K - 0.5)
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"state {k}" for k in range(K)])
    ax.set_xlabel("time (s)")
    ax.set_title(title or "Discrete-state runs")

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    plt.show()
