# ── src/single_rslds.py ──────────────────────────────────────────────
import os, pickle, h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import ssm                       # pip install ssm==0.0.1 or your fork

__all__ = ["fit_single_rslds"]

def _footshock_vector(t, shock_times):
    v = np.zeros_like(t, dtype=float)
    idx = np.searchsorted(t, shock_times)
    idx = np.clip(idx, 0, len(t)-1)
    v[idx] = 1.0
    return v[:, None]            # (T,1)

def _auto_latent_dim(fr_z, variance_goal=0.90, cap=30):
    pcs = PCA().fit(fr_z)
    cum = np.cumsum(pcs.explained_variance_ratio_)
    d   = np.argmax(cum >= variance_goal) + 1
    return int(min(d, cap))

def fit_single_rslds(h5_path,
                     csv_path,
                     save_dir,
                     *,
                     variance_goal = 0.90,   # ignored if latent_dim given
                     latent_dim    = None,
                     K_states      = 2,
                     num_iters     = 300,
                     overwrite     = False,
                     verbose       = True):
    """
    Fit an input-driven rSLDS to one rat × area dataset.

    h5_path : the raw HDF5 (needs 'time' and 'footshock_times')
    csv_path: the wide CSV (time_s, speed, neuron columns)
    save_dir: folder where all files will be written
    """

    save_dir = Path(save_dir)
    if save_dir.exists() and not overwrite:
        raise FileExistsError(f"{save_dir} exists (use overwrite=True)")
    save_dir.mkdir(parents=True, exist_ok=True)

    # -------- load CSV
    df  = pd.read_csv(csv_path)
    t   = df.pop("time_s").values
    speed = df.pop("speed").values
    FR  = df.values.astype(float)            # (T, N)

    # -------- load foot-shock times
    with h5py.File(h5_path, "r") as h5:
        shock_times = h5["footshock_times"][...]

    u = _footshock_vector(t, shock_times)    # (T,1)

    # -------- z-score firing rates
    mu = FR.mean(0, keepdims=True)
    sd = FR.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    FR_z = (FR - mu) / sd

    # -------- choose latent dimension
    if latent_dim is None:
        latent_dim = _auto_latent_dim(FR_z, variance_goal)
        if verbose:
            print(f"latent_dim auto→ {latent_dim}")

    # -------- build & fit rSLDS
    model = ssm.SLDS(
        D_obs = FR_z.shape[1],
        K     = K_states,
        D_latent = latent_dim,
        M     = 1,
        transitions = "inputdriven",
        dynamics    = "gaussian",
        emissions   = "ar",
        single_subspace = True
    )

    elbos, post = model.fit(
        [FR_z], inputs=[u],
        num_iters=num_iters,
        method="bbvi",
        variational_posterior="meanfield"
    )

    x_hat = post.mean[0]
    z_hat = model.most_likely_states(x_hat, FR_z)

    # -------- save artefacts
    # whole model
    with open(save_dir/"rSLDS.pkl", "wb") as f: pickle.dump(model, f)
    # key components for inspection
    with open(save_dir/"emissions.pkl",  "wb") as f: pickle.dump(model.emissions,  f)
    with open(save_dir/"transitions.pkl","wb") as f: pickle.dump(model.transitions, f)
    with open(save_dir/"dynamics.pkl",  "wb") as f: pickle.dump(model.dynamics,   f)

    # numpy arrays
    np.save(save_dir/"elbos.npy", elbos)
    np.save(save_dir/"x_hat.npy", x_hat)
    np.save(save_dir/"z_hat.npy", z_hat)
    np.save(save_dir/"FR_z.npy", FR_z)
    np.save(save_dir/"footshock.npy", u)
    np.save(save_dir/"speed.npy", speed)

    # ELBO plot
    plt.figure()
    plt.plot(elbos)
    plt.xlabel("iteration"); plt.ylabel("ELBO")
    plt.title(f"rSLDS fit  (K={K_states}, Lat={latent_dim})")
    plt.tight_layout()
    plt.savefig(save_dir/"elbo.png")
    plt.close()

    if verbose:
        print(f"✓ rSLDS trained and saved → {save_dir}")

    return dict(model=model, elbos=elbos, x_hat=x_hat,
                z_hat=z_hat, save_dir=str(save_dir))
