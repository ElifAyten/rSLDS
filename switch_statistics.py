
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

def plot_switch_summary2(z_states, time_vec, footshock_mask, *, figsize=(15, 4)):
    """Three clearly separated panels: switch count, rate, and occupancy."""

    s = switch_statistics(z_states, time_vec, footshock_mask)
    K = s["K"]

    C_PRE  = "#2166AC"   # blue   (ColorBrewer RdBu)
    C_POST = "#B2182B"   # red

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=0.5)  # <- increase horizontal space between plots

    # (1) switch count
    axs[0].bar(["pre", "post"],
               [s["n_switch_pre"], s["n_switch_post"]],
               color=[C_PRE, C_POST])
    axs[0].set_title("Switch Count")
    axs[0].set_ylabel("Count")
    for i, v in enumerate([s["n_switch_pre"], s["n_switch_post"]]):
        axs[0].text(i, v, f"{v}", ha="center", va="bottom")

    # (2) switch rate
    axs[1].bar(["pre", "post"],
               [s["rate_pre"], s["rate_post"]],
               color=[C_PRE, C_POST])
    axs[1].set_title("Switches / Minute")
    axs[1].set_ylabel("Rate")
    for i, v in enumerate([s["rate_pre"], s["rate_post"]]):
        axs[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")

    # (3) state occupancy
    x, w = np.arange(K), 0.35
    axs[2].bar(x - w/2, s["occupancy_pre"],  width=w, label="pre",  color=C_PRE)
    axs[2].bar(x + w/2, s["occupancy_post"], width=w, label="post", color=C_POST)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([f"s{k}" for k in x])
    axs[2].set_title("State Occupancy")
    axs[2].set_ylabel("Fraction")
    axs[2].legend(frameon=False)

    for k in x:
        axs[2].text(k - w/2, s["occupancy_pre"][k],
                    f"{s['occupancy_pre'][k]:.1%}", ha="center", va="bottom", fontsize=8)
        axs[2].text(k + w/2, s["occupancy_post"][k],
                    f"{s['occupancy_post'][k]:.1%}", ha="center", va="bottom", fontsize=8)

    for ax in axs:
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)

    plt.show()
    return s

