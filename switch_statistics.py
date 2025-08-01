import numpy as np
import matplotlib.pyplot as plt

def switch_statistics(z_states, time_vec, footshock_mask):
    z_states = np.asarray(z_states, int)
    time_vec = np.asarray(time_vec, float)
    footshock_mask = np.asarray(footshock_mask, bool)

    first_idx = np.where(footshock_mask)[0][0]
    t0 = time_vec[first_idx]

    pre  = time_vec <  t0
    post = time_vec >= t0

    switches = np.diff(z_states) != 0
    n_switch_pre  = switches[pre[1:]].sum()
    n_switch_post = switches[post[1:]].sum()

    dur_pre  = time_vec[pre][-1]  - time_vec[pre][0]
    dur_post = time_vec[post][-1] - time_vec[post][0]
    rate_pre  = n_switch_pre  / (dur_pre  / 60.0) if dur_pre  > 0 else np.nan
    rate_post = n_switch_post / (dur_post / 60.0) if dur_post > 0 else np.nan

    K = z_states.max() + 1
    occ_pre  = np.bincount(z_states[pre],  minlength=K) / pre.sum()
    occ_post = np.bincount(z_states[post], minlength=K) / post.sum()

    return {
        "n_switch_pre": int(n_switch_pre),
        "n_switch_post": int(n_switch_post),
        "rate_pre": rate_pre,
        "rate_post": rate_post,
        "occupancy_pre": occ_pre,
        "occupancy_post": occ_post,
        "K": K,
        "t0": t0,
    }

# --- compute stats for current data --------------------------------------
stats = switch_statistics(z_states, time_vec, footshock_vec)
K = stats["K"]

# 1) Switch count bar chart
plt.figure(figsize=(4,3))
plt.bar(["pre", "post"], [stats["n_switch_pre"], stats["n_switch_post"]])
plt.ylabel("Number of switches")
plt.title("Switch count before vs after first shock")
plt.tight_layout()
plt.show()

# 2) Switch rate bar chart
plt.figure(figsize=(4,3))
plt.bar(["pre", "post"], [stats["rate_pre"], stats["rate_post"]])
plt.ylabel("Switches per minute")
plt.title("Switch rate before vs after first shock")
plt.tight_layout()
plt.show()

# 3) State occupancy grouped bar chart
x = np.arange(K)
width = 0.35
plt.figure(figsize=(1.2*K + 3, 3))
plt.bar(x - width/2, stats["occupancy_pre"], width, label="pre")
plt.bar(x + width/2, stats["occupancy_post"], width, label="post")
plt.xticks(x, [f"state {k}" for k in x])
plt.ylabel("Fraction of samples")
plt.title("State occupancy before vs after first shock")
plt.legend()
plt.tight_layout()
plt.show()
