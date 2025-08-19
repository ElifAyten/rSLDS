
# Visualise each discrete state's linear dynamical flow in the top two
# principal‑component dimensions of an rSLDS latent trajectory.
import glob
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


__all__ = ["plot_rslds_vector_field"]


def _grab(pattern: str | Path):
    """Return the first file that matches *pattern*, raise if none."""
    matches = glob.glob(str(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matches {pattern}")
    return matches[0]


def plot_rslds_vector_field(
    model_dir: str | Path,
    *,
    n_components: int = 2,
    grid_size: int = 25,
    density: float = 1.2,
    cmap: str = "viridis",
    point_size: int = 6,
    alpha_points: float = 0.3,
    figsize_base: float = 3,
):
    """
    Visualise the linear flow field of each discrete state in a PCA plane.

    Parameters
    ----------
    model_dir : str or Path
        Directory that contains
            * x_hat*.npy   – latent trajectories (T × D)
            * z_hat*.npy   – discrete state labels (T,)
            * *model*.pkl  – the fitted rSLDS object
    """
    model_dir = Path(model_dir)

    x_hat_path = _grab(model_dir / "x_hat*.npy")
    z_hat_path = _grab(model_dir / "z_hat*.npy")
    pkl_path   = _grab(model_dir / "*model*.pkl")

    # load data
    x_hat = np.load(x_hat_path, allow_pickle=True)
    z_hat = np.load(z_hat_path, allow_pickle=True)
    with open(pkl_path, "rb") as fh:
        model = pickle.load(fh)

    # pull dynamics matrices
    As = model.dynamics.As       # shape (K, D, D)
    bs = model.dynamics.bs       # shape (K, D)
    K, D_latent, _ = As.shape

    # PCA projection to 2‑D 
    pca = PCA(n_components=n_components)
    x2  = pca.fit_transform(x_hat)        # (T, 2)
    P, mu = pca.components_, pca.mean_    # (2, D), (D,)

    # project each state's A,b into PCA plane
    A2_list, b2_list = [], []
    for k in range(K):
        A, b = As[k], bs[k]
        A2   = P @ A @ P.T
        b2   = P @ (A @ mu + b - mu)
        A2_list.append(A2)
        b2_list.append(b2)

    # build a grid
    xmin, xmax = x2[:, 0].min(), x2[:, 0].max()
    ymin, ymax = x2[:, 1].min(), x2[:, 1].max()
    pad_x, pad_y = 0.2 * (xmax - xmin), 0.2 * (ymax - ymin)
    xs = np.linspace(xmin - pad_x, xmax + pad_x, grid_size)
    ys = np.linspace(ymin - pad_y, ymax + pad_y, grid_size)
    X, Y = np.meshgrid(xs, ys)

    # plot
    fig, axes = plt.subplots(
        1, K, figsize=(figsize_base * K, figsize_base), sharex=True, sharey=True
    )
    if K == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(grid_size):
            for j in range(grid_size):
                pt  = np.array([X[i, j], Y[i, j]])
                dpt = (A2_list[k] - np.eye(2)) @ pt + b2_list[k]
                U[i, j], V[i, j] = dpt

        speed = np.sqrt(U ** 2 + V ** 2)
        ax.streamplot(
            X, Y, U, V, color=speed, cmap=cmap, density=density, linewidth=1
        )

        mask = z_hat == k
        ax.scatter(
            x2[mask, 0],
            x2[mask, 1],
            s=point_size,
            alpha=alpha_points,
            c="black",
        )

        ax.set_title(f"state {k}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)

    fig.tight_layout()
    plt.show()

