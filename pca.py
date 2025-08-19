# rSLDS/pca.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA


__all__ = ["pca_summary", "plot_pca_cumsum"]

def pca_summary(data, *, var_targets=(0.90, 0.95), elbow_drop=0.01):
    """
    Run PCA on a (time × neurons) matrix and report useful thresholds.

    Parameters
    ----------
    data : ndarray  shape (T, N) or (N, T)
        Firing-rate matrix. If rows > cols, it is transposed automatically.
    var_targets : tuple of float
        Cumulative-variance percentages you care about (default 90 %, 95 %).
    elbow_drop : float
        First-derivative threshold that defines the 'elbow' (default 0.01).

    Returns
    -------
    summary : dict
        {
          'explained' : 1-D array of cumulative variance,
          'elbow'     : k,
          '90%'       : k90,
          '95%'       : k95,
          ...
        }
        One key per var_target.
    """
    X = data if data.shape[0] < data.shape[1] else data.T


    pca = PCA()
    pca.fit(X)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    summary = {"explained": cumsum}

    # variance cut-offs
    for vt in var_targets:
        k = np.argmax(cumsum >= vt) + 1
        summary[f"{int(vt*100)}%"] = k

    # elbow: first point where marginal gain < elbow_drop
    diff = np.diff(cumsum, prepend=0)
    elbow = np.argmax(diff < elbow_drop) + 1
    summary["elbow"] = elbow

    return summary


def plot_pca_cumsum(summary, *, ax=None, title="PCA cumulative variance"):
    """
    Visualise cumulative explained variance and annotate thresholds.

    Parameters
    ----------
    summary : dict
        Output of `pca_summary`.
    ax : matplotlib Axes or None
        Pass an existing Axes to draw on, or get a new figure if None.

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    cumsum = summary["explained"]
    ax.plot(np.arange(1, len(cumsum)+1), cumsum, marker="o", ls="--")

    # annotate targets
    colors = {"90%": "g", "95%": "b", "elbow": "r"}
    for key, col in colors.items():
        if key in summary:
            if key == "elbow":
                ax.axvline(summary[key], color=col, ls="--",
                           label=f"elbow: {summary[key]} comps")
            else:
                ax.axhline(int(key.rstrip('%'))/100, color=col, ls=":",
                           label=f"{key} at {summary[key]} comps")

    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="lower right")
    return ax
