from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mutual_information as mi

# ensure mi has a plt handle (for older modules that expect it)
if not hasattr(mi, "plt"):
    mi.plt = plt

def mi_pre_post_plot(
    *,
    latents: np.ndarray,
    signal: np.ndarray,
    time_vec: np.ndarray,
    footshock_mask: np.ndarray,
    win_s: float            = 0.3,
    integrate_latents: bool = False,
    shuffle: str            = "circular",
    n_shuffle: int          = 500,
    random_state: int       = 0,
    color_pre: str          = "#3B5BA9",
    color_post: str         = "#C43B3B",
    figsize                 = None,
    csv_path: str | None    = None,
    # NEW:
    fdr_alpha: float | None = 0.05,
):
    """
    Run MI analysis pre- vs post-shock and return (table, fig).
    If your mutual_information.compare_mi_pre_post supports `return_rich`
    and `fdr_alpha`, we’ll use the “rich” outputs (p_perm, BH-FDR, z, Δ vs null).
    Otherwise, we gracefully fall back to the simpler outputs.
    """
    sig = np.nan_to_num(signal, nan=np.nanmean(signal))

    # Try rich call first; if the installed compare_mi_pre_post is older,
    # fall back to the legacy signature.
    try:
        pre, post = mi.compare_mi_pre_post(
            latents           = latents,
            signal            = sig,
            time_vec          = time_vec,
            footshock_mask    = footshock_mask,
            win_s             = win_s,
            integrate_latents = integrate_latents,
            shuffle           = shuffle,
            n_shuffle         = n_shuffle,
            random_state      = random_state,
            plot              = False,
            return_rich       = True,
            fdr_alpha         = fdr_alpha,
        )
        rich = True
    except TypeError:
        pre, post = mi.compare_mi_pre_post(
            latents           = latents,
            signal            = sig,
            time_vec          = time_vec,
            footshock_mask    = footshock_mask,
            win_s             = win_s,
            integrate_latents = integrate_latents,
            shuffle           = shuffle,
            n_shuffle         = n_shuffle,
            random_state      = random_state,
            plot              = False,
        )
        rich = False

    # Build the table depending on what fields we have
    if rich and all(name in pre.dtype.names for name in
                    ("MI_null_mean","MI_null_std","z","p_perm","q_bh","reject_bh","significant_95")):
        table = pd.DataFrame({
            "dim"            : pre["dim"],
            "MI_pre"         : pre["MI_raw"],
            "MI_post"        : post["MI_raw"],
            "ΔMI"            : post["MI_raw"] - pre["MI_raw"],
            "thr95_pre"      : pre["thr95"],
            "thr95_post"     : post["thr95"],
            "MI_mu_pre"      : pre["MI_null_mean"],
            "MI_mu_post"     : post["MI_null_mean"],
            "MI_sd_pre"      : pre["MI_null_std"],
            "MI_sd_post"     : post["MI_null_std"],
            "z_pre"          : pre["z"],
            "z_post"         : post["z"],
            "p_perm_pre"     : pre["p_perm"],
            "p_perm_post"    : post["p_perm"],
            "q_bh_pre"       : pre["q_bh"],
            "q_bh_post"      : post["q_bh"],
            "reject_bh_pre"  : pre["reject_bh"],
            "reject_bh_post" : post["reject_bh"],
            "sig95_pre"      : pre["significant_95"],
            "sig95_post"     : post["significant_95"],
        })
        star_mode = "bh_or_thr95"  # prefer BH-FDR, else 95th percentile
    else:
        # Legacy simple dtype: ('dim','MI_raw','MI_min','thr95','p')
        table = pd.DataFrame({
            "dim"            : pre["dim"],
            "MI_pre"         : pre["MI_raw"],
            "MI_post"        : post["MI_raw"],
            "ΔMI"            : post["MI_raw"] - pre["MI_raw"],
            "thr95_pre"      : pre["thr95"],
            "thr95_post"     : post["thr95"],
            "p_pre"          : pre["p"],
            "p_post"         : post["p"],
        })
        star_mode = "thr95_or_p"   # draw star if raw > thr95 or p<.05

    if csv_path:
        table.to_csv(csv_path, index=False)

    # Plot
    dims  = table["dim"].to_numpy()
    x     = np.arange(len(dims))
    w     = 0.35
    if figsize is None:
        figsize = (0.85 * len(dims) + 3, 3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - w/2, table["MI_pre"],  w, color=color_pre,  label="pre-shock")
    ax.bar(x + w/2, table["MI_post"], w, color=color_post, label="post-shock")

    # Stars for POST bars
    if star_mode == "bh_or_thr95":
        for xi, raw, rej_bh, thr95 in zip(x, table["MI_post"], table["reject_bh_post"], table["thr95_post"]):
            if (fdr_alpha is not None and bool(rej_bh)) or (fdr_alpha is None and raw > thr95):
                ax.text(xi + w/2, raw, "★", ha="center", va="bottom",
                        color=color_post, fontsize=10)
        subtitle = f"BH q≤{fdr_alpha}" if fdr_alpha is not None else "95% null"
    else:
        for xi, raw, thr95, pval in zip(x, table["MI_post"], table["thr95_post"], table["p_post"]):
            if (raw > thr95) or (np.isfinite(pval) and pval < 0.05):
                ax.text(xi + w/2, raw, "★", ha="center", va="bottom",
                        color=color_post, fontsize=10)
        subtitle = "p<.05 or >95% null"

    ax.set_xticks(x)
    ax.set_xticklabels([f"dim {d}" for d in dims])
    ax.set_ylabel("Mutual information")
    ax.set_title(
        f"Pre- vs post-shock MI ({int(win_s*1e3)} ms windows, "
        f'{"cum" if integrate_latents else "raw"} latents; {subtitle})'
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return table, fig

