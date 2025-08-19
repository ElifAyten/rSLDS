import os, pickle, h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import ssm                                   # pip install ssm==0.0.1

__all__ = ["fit_single_rslds"]

def _footshock_vector(t, shock_times):
    v = np.zeros_like(t, dtype=float)
    idx = np.searchsorted(t, shock_times)
    idx = np.clip(idx, 0, len(t) - 1)
    v[idx] = 1.0
    return v[:, None]                         # (T, 1)


def _auto_latent_dim(fr, variance_goal=0.90, cap=30):
    pcs = PCA().fit(fr)
    cum = np.cumsum(pcs.explained_variance_ratio_)
    d = np.argmax(cum >= variance_goal) + 1
    return int(min(d, cap))


def fit_single_rslds(
    h5_path,
    csv_path,
    save_dir,
    *,
    variance_goal=0.90,
    latent_dim=None,
    kappa=0.0,                
    K_states=2,
    num_iters=300,
    overwrite=False,
    verbose=True,
):
    """
    Fit an input-driven rSLDS to one rat × area dataset.

    Parameters
    ----------
    h5_path  : raw HDF5 file (contains 'time' and 'footshock_times')
    csv_path : *wide* CSV  (time_s, [speed], 1 col per neuron)
    save_dir : folder to write model & artefacts
    """

    #  output folder
    save_dir = Path(save_dir)
    if save_dir.exists() and not overwrite:
        raise FileExistsError(f"{save_dir} exists (use overwrite=True)")
    save_dir.mkdir(parents=True, exist_ok=True)

    # load CSV
    df = pd.read_csv(csv_path)
    t = df.pop("time_s").values

    if "speed" in df.columns:
        speed = df.pop("speed").values
        if verbose: print("✓ speed column found in CSV")
    else:
        with h5py.File(h5_path, "r") as h5:
            speed = h5["speed"][...]
        if verbose: print("speed column missing → loaded from HDF5")

    FR = df.values.astype(float)              # (T, N_neurons)

    # foot-shock regressor 
    with h5py.File(h5_path, "r") as h5:
        shock_times = h5["footshock_times"][...]
    u = _footshock_vector(t, shock_times)     # (T, 1)

    # z-score firing rates 
    mu, sd = FR.mean(0, keepdims=True), FR.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    FR_z = (FR - mu) / sd

    # pick latent dimension 
    if latent_dim is None:
        latent_dim = _auto_latent_dim(FR, variance_goal)
        if verbose: print(f"latent_dim auto → {latent_dim}")

    # build rSLDS model
    model = ssm.SLDS(
        FR_z.shape[1],             # D_obs (neurons)
        K_states,                  # K (discrete states)
        latent_dim,                # D_latent
        M=1,                       # one exogenous input (shock)
        transitions="inputdriven",
        transition_kwargs=dict(kappa=kappa),   # ← stickiness
        dynamics="gaussian",
        emissions="ar",
        single_subspace=True,
    )

    # fit with variational inference 
    elbos, post = model.fit(
        [FR_z],
        inputs=[u],
        num_iters=num_iters,
        method="bbvi",
        variational_posterior="meanfield",
    )

    x_hat = post.mean[0]                          # (T, latent_dim)
    z_hat = model.most_likely_states(x_hat, FR_z)

    # save artefacts
    with open(save_dir / "rSLDS.pkl", "wb") as f:
        pickle.dump(model, f)

    for name, obj in [("emissions", model.emissions),
                      ("transitions", model.transitions),
                      ("dynamics", model.dynamics)]:
        with open(save_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    np.save(save_dir / "elbos.npy", elbos)
    np.save(save_dir / "x_hat.npy",  x_hat)
    np.save(save_dir / "z_hat.npy",  z_hat)
    np.save(save_dir / "FR_z.npy",   FR_z)
    np.save(save_dir / "footshock.npy", u)
    np.save(save_dir / "speed.npy",  speed)

    #  ELBO convergence plot
    #elbo is the models score during training bigger is better (often it’s a negative number—less negative = better)
    plt.figure()
    plt.plot(elbos)
    plt.xlabel("iteration")
    plt.ylabel("ELBO")
    plt.title(f"rSLDS fit  (K={K_states}, Latent={latent_dim}, κ={kappa})")
    plt.tight_layout()
    plt.savefig(save_dir / "elbo.png")
    plt.close()

    if verbose:
        print(f"✓ rSLDS trained and saved → {save_dir}")

    return dict(
        model   = model,
        elbos   = elbos,
        x_hat   = x_hat,
        z_hat   = z_hat,
        save_dir= str(save_dir),
    )
