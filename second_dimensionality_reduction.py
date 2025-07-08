import os, pickle, numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA

__all__ = ["plot_rslds_vector_field"]

def plot_rslds_vector_field(
    model_dir,
    *,
    n_components = 2,
    grid_size    = 25,
    density      = 1.2,
    cmap         = "viridis",
    point_size   = 6,
    alpha_points = 0.3,
    figsize_base = 3
):
    """
    Visualise each discrete state's linear flow field in a PCA plane.

    Parameters
    model_dir : str | Path
        Folder that contains x_hat_*.npy, z_hat_*.npy, and slDS_model_*.pkl.
    n_components : int
        How many PCs to project onto (only 2 is supported for plotting).
    grid_size : int
        Number of grid points per dimension for streamplot.
    density : float
        Streamplot density parameter.
    """
    model_dir = os.fspath(model_dir)

    # load data 
    x_hat = np.load([f for f in os.listdir(model_dir)
                     if f.startswith("x_hat")][0], allow_pickle=True)
    z_hat = np.load([f for f in os.listdir(model_dir)
                     if f.startswith("z_hat")][0], allow_pickle=True)

    pkl_path = [f for f in os.listdir(model_dir)
                if f.startswith("slDS_model") and f.endswith(".pkl")][0]
    with open(os.path.join(model_dir, pkl_path), "rb") as f:
        model = pickle.load(f)

    As = model.dynamics.As     # (K,D,D)
    bs = model.dynamics.bs     # (K,D)
    K, D_latent, _ = As.shape

    # PCA 
    pca = PCA(n_components=n_components)
    x2  = pca.fit_transform(x_hat)  # (T, 2)
    P   = pca.components_          # (2, D_latent)
    mu  = pca.mean_                # (D_latent,)

    # project each state's A,b
    A2_list, b2_list = [], []
    for k in range(K):
        A = As[k]; b = bs[k]
        A2 = P @ A @ P.T
        b2 = P @ (A @ mu + b - mu)
        A2_list.append(A2); b2_list.append(b2)

    # grid over PCA space
    xmin, xmax = x2[:,0].min(), x2[:,0].max()
    ymin, ymax = x2[:,1].min(), x2[:,1].max()
    pad_x, pad_y = .2*(xmax-xmin), .2*(ymax-ymin)
    xs = np.linspace(xmin-pad_x, xmax+pad_x, grid_size)
    ys = np.linspace(ymin-pad_y, ymax+pad_y, grid_size)
    X, Y = np.meshgrid(xs, ys)

    # plot
    fig, axes = plt.subplots(1, K, figsize=(figsize_base*K, figsize_base),
                             sharex=True, sharey=True)
    if K == 1: axes = [axes]

    for k, ax in enumerate(axes):
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(grid_size):
            for j in range(grid_size):
                pt   = np.array([X[i,j], Y[i,j]])
                dpt  = (A2_list[k] - np.eye(2)) @ pt + b2_list[k]
                U[i,j], V[i,j] = dpt

        speed = np.sqrt(U**2 + V**2)
        ax.streamplot(X, Y, U, V, color=speed,
                      cmap=cmap, density=density, linewidth=1)

        # overlay latent points of this state
        mask = (z_hat == k)
        ax.scatter(x2[mask,0], x2[mask,1],
                   s=point_size, alpha=alpha_points, c="black")

        ax.set_title(f"state {k}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

        ax.set_xlim(xmin-pad_x, xmax+pad_x)
        ax.set_ylim(ymin-pad_y, ymax+pad_y)

    fig.tight_layout()
    plt.show()
