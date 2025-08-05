
import numpy as np
import matplotlib.pyplot as plt

# ---------------- switch_statistics (unchanged) --------------------------
def switch_statistics(z_states, time_vec, footshock_mask):
    z_states       = np.asarray(z_states,    int)
    time_vec       = np.asarray(time_vec,    float)
    footshock_mask = np.asarray(footshock_mask, bool)

    first_idx = np.where(footshock_mask)[0][0]
    t0 = time_vec[first_idx]

    pre   = time_vec <  t0
    post  = time_vec >= t0
    dt    = np.diff(time_vec, prepend=time_vec[0])   # sample spacing

    # number of switches
    switches       = np.diff(z_states) != 0
    n_switch_pre   = switches[pre[1:]].sum()
    n_switch_post  = switches[post[1:]].sum()

    # recording length (min) for rate
    dur_pre_min    = dt[pre].sum()  / 60.0
    dur_post_min   = dt[post].sum() / 60.0
    rate_pre       = n_switch_pre  / dur_pre_min   if dur_pre_min  > 0 else np.nan
    rate_post      = n_switch_post / dur_post_min  if dur_post_min > 0 else np.nan

    # occupancy
    K = z_states.max() + 1
    occ_pre  = np.bincount(z_states[pre],  minlength=K) / pre.sum()
    occ_post = np.bincount(z_states[post], minlength=K) / post.sum()

    return dict(
        n_switch_pre   = int(n_switch_pre),
        n_switch_post  = int(n_switch_post),
        rate_pre       = rate_pre,
        rate_post      = rate_post,
        occupancy_pre  = occ_pre,
        occupancy_post = occ_post,
        K              = K,
        t0             = t0,
    )

import numpy as np
import matplotlib.pyplot as plt

def plot_switch_summary2(z_states, time_vec, footshock_mask, *,
                         figsize=(15, 4), separate=False, dpi=100):
    """
    Show switch count / rate / occupancy.

    Parameters
    ----------
    separate : bool
        False  → return single 3-panel figure (original behaviour)
        True   → return list of three single-panel figures
    """
    s = switch_statistics(z_states, time_vec, footshock_mask)
    K = s["K"]
    C_PRE, C_POST = "#2166AC", "#B2182B"

    def _make_ax(tag):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=dpi)
        if tag == "count":
            vals = [s["n_switch_pre"], s["n_switch_post"]]
            ax.bar(["pre", "post"], vals, color=[C_PRE, C_POST])
            ax.set_title("Switch count")
            for i, v in enumerate(vals):
                ax.text(i, v, str(v), ha="center", va="bottom")
        elif tag == "rate":
            vals = [s["rate_pre"], s["rate_post"]]
            ax.bar(["pre", "post"], vals, color=[C_PRE, C_POST])
            ax.set_title("Switches / minute")
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
        else:  # occupancy
            x, w = np.arange(K), 0.35
            ax.bar(x-w/2, s["occupancy_pre"],  w, label="pre",  color=C_PRE)
            ax.bar(x+w/2, s["occupancy_post"], w, label="post", color=C_POST)
            ax.set_xticks(x); ax.set_xticklabels([f"s{k}" for k in x])
            ax.set_title("State occupancy"); ax.legend(frameon=False)
            for k in x:
                ax.text(k-w/2, s["occupancy_pre"][k],
                        f"{s['occupancy_pre'][k]:.1%}", ha="center", va="bottom", fontsize=8)
                ax.text(k+w/2, s["occupancy_post"][k],
                        f"{s['occupancy_post'][k]:.1%}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(bottom=0); ax.spines[["top", "right"]].set_visible(False)
        return fig, ax

    if separate:
        figs = [_make_ax(tag)[0] for tag in ("count", "rate", "occ")]
        plt.show()    # display them inline
        return s, figs

    # combined figure (original behaviour)
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    fig.subplots_adjust(wspace=0.5)
    for ax, tag in zip(axs, ("count", "rate", "occupancy")):
        tmp_fig, tmp_ax = _make_ax(tag if tag != "occupancy" else "occ")
        for artist in tmp_ax.get_children():
            artist.figure = fig
            ax._add_text(artist) if isinstance(artist, plt.Text) else ax.add_artist(artist)
        plt.close(tmp_fig)
    plt.show()
    return s, [fig]
