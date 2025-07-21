import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_rslds_vector_field_corrected(
    model_dir,
    *,
    n_components = 2,
    grid_size    = 25,
    density      = 1.2,
    cmap         = "viridis",
    point_size   = 6,
    alpha_points = 0.3,
    figsize_base = 3,
):
    """
    Visualise each discrete state's linear flow field in a PCA plane.
    """
    model_dir = os.fspath(model_dir)

    # ── load latent arrays (corrected) ───────────────────────────
    x_hat_path = os.path.join(model_dir, "x_hat.npy")
    z_hat_path = os.path.join(model_dir, "z_hat.npy")

    x_hat = np.load(x_hat_path, allow_pickle=True)
    z_hat = np.load(z_hat_path, allow_pickle=True)

    # ── locate the model pickle (new logic) ─────────────────────────
    pkl_candidates = [f for f in os.listdir(model_dir)
                      if f.endswith(".pkl")
                      and not f.startswith(("transitions", "dynamics", "emissions"))]
    if not pkl_candidates:
        raise FileNotFoundError("no model *.pkl found in model_dir")
    pkl_path = sorted(pkl_candidates)[0]          # first match: slDS_model*.pkl or rSLDS.pkl

    with open(os.path.join(model_dir, pkl_path), "rb") as f:
        model = pickle.load(f)

    # ── rest of the function (identical) ────────────────────────────
    As = model.dynamics.As     # (K,D,D)
    bs = model.dynamics.bs     # (K,D)
    K, D_latent, _ = As.shape

    pca = PCA(n_components=n_components)
    x2  = pca.fit_transform(x_hat)
    P, mu = pca.components_, pca.mean_

    A2_list, b2_list = [], []
    for k in range(K):
        A, b = As[k], bs[k]
        A2 = P @ A @ P.T
        b2 = P @ (A @ mu + b - mu)
        A2_list.append(A2); b2_list.append(b2)

    xmin, xmax = x2[:,0].min(), x2[:,0].max()
    ymin, ymax = x2[:,1].min(), x2[:,1].max()
    pad_x, pad_y = .2*(xmax-xmin), .2*(ymax-ymin)
    xs = np.linspace(xmin-pad_x, xmax+pad_x, grid_size)
    ys = np.linspace(ymin-pad_y, ymax+pad_y, grid_size)
    X, Y = np.meshgrid(xs, ys)

    fig, axes = plt.subplots(1, K, figsize=(figsize_base*K, figsize_base),
                             sharex=True, sharey=True)
    if K == 1: axes = [axes]

    for k, ax in enumerate(axes):
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(grid_size):
            for j in range(grid_size):
                pt  = np.array([X[i,j], Y[i,j]])
                dpt = (A2_list[k] - np.eye(2)) @ pt + b2_list[k]
                U[i,j], V[i,j] = dpt

        speed = np.sqrt(U**2 + V**2)
        ax.streamplot(X, Y, U, V, color=speed,
                      cmap=cmap, density=density, linewidth=1)

        mask = (z_hat == k)
        ax.scatter(x2[mask,0], x2[mask,1],
                   s=point_size, alpha=alpha_points, c="black")

        ax.set_title(f"state {k}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xlim(xmin-pad_x, xmax+pad_x)
        ax.set_ylim(ymin-pad_y, ymax+pad_y)

    fig.tight_layout()
    plt.show()

# Call the corrected function
model_dir = "/content/drive/My Drive/rSLD 3/rSLD/models_Rat4_ventral"
plot_rslds_vector_field_corrected(
    model_dir=model_dir,
    grid_size=30,
    cmap="plasma",
    density=1.0
)
