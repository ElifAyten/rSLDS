from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
import ssm

def footshock_vector(time_vec, shock_times):
    """
    Binary regressor aligned to time_vec. Safe if shock_times is None/empty.
    """
    t = np.asarray(time_vec).ravel()
    v = np.zeros_like(t, dtype=float)
    if shock_times is None or len(shock_times) == 0:
        return v[:, None]
    shock_times = np.asarray(shock_times).ravel()
    idx = np.searchsorted(t, shock_times)
    idx = idx[(idx >= 0) & (idx < len(t))]
    v[idx] = 1.0
    return v[:, None]

def _auto_latent_dim_pca(train_Z: np.ndarray, variance_goal: float = 0.90, cap: int = 30):

    if not (0.0 < float(variance_goal) <= 1.0):
        raise ValueError("variance_goal must be in (0,1], e.g., 0.90")
    # Fit PCA on TRAIN only
    pca = PCA(svd_solver="full")
    pca.fit(train_Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    d = int(np.searchsorted(cum, variance_goal) + 1)
    d = max(1, min(d, min(cap, train_Z.shape[1])))
    achieved = float(cum[d-1]) if d <= len(cum) else float(cum[-1])
    return d, achieved, pca

def crossval_rslds(
    h5_path,
    csv_path,
    save_dir,
    *,
    K_states=4,
    num_iters=300,
    kappa=0.0,
    n_folds=5,
    variance_goal=0.90,
    verbose=True
):
    """
    K-Fold cross-validation for rSLDS with TRAIN-only PCA to reach variance_goal.
    Returns a list of dicts (one per fold) with model & ELBOs.
    """
    # load wide CSV
    df = pd.read_csv(csv_path)
    t = df.pop("time_s").values
    FR = df.values.astype(float)
    FR = np.nan_to_num(FR, nan=0.0)

    # shock regressor from HDF5 (safe if missing)
    with h5py.File(h5_path, "r") as h5:
        shock_times = h5["footshock_times"][...].ravel() if "footshock_times" in h5 else None
    u = footshock_vector(t, shock_times)  # (T,1)

    # CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for i, (train_idx, test_idx) in enumerate(kf.split(FR)):
        if verbose:
            print(f"=== Fold {i+1}/{n_folds} (train {len(train_idx)}, test {len(test_idx)}) ===")

        # split raw FR & input
        FR_tr_raw, FR_te_raw = FR[train_idx], FR[test_idx]
        u_tr, u_te = u[train_idx], u[test_idx]

        # z-score using TRAIN stats only
        mu = FR_tr_raw.mean(0, keepdims=True)
        sd = FR_tr_raw.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        FR_tr = FR_tr_raw
        FR_te = FR_te_raw    
        
        """ depending on what we want we can also z-score here """


        latent_dim_i, achieved, pca_obj = _auto_latent_dim_pca(FR_tr, variance_goal, cap=30)
        if verbose:
            print(f"[auto/PCA] variance_goal={variance_goal:.2f} → latent_dim={latent_dim_i} "
                  f"(achieved cum var={achieved:.3f})")

        # fit model on TRAIN
        model = ssm.SLDS(
            FR_tr.shape[1],           # D_obs
            K_states,                 # K
            latent_dim_i,             # D_latent
            M=1,                      # one exogenous input (shock)
            transitions="inputdriven",
            transition_kwargs=dict(kappa=kappa),
            dynamics="gaussian",
            emissions="ar",
            single_subspace=True,
        )
        elbos, post = model.fit(
            [FR_tr],
            inputs=[u_tr],
            num_iters=num_iters,
            method="bbvi",
            variational_posterior="meanfield",
        )

        train_elbo = float(elbos[-1])

        # save outputs per-fold
        fold_dir = Path(save_dir) / f"cv_fold{i+1}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        np.save(fold_dir / "elbos.npy", np.asarray(elbos))
        with open(fold_dir / "meta.txt", "w") as f:
            f.write(
                f"K_states={K_states}\n"
                f"kappa={kappa}\n"
                f"num_iters={num_iters}\n"
                f"latent_dim={latent_dim_i}\n"
                f"achieved_cum_variance={achieved:.6f}\n"
                f"n_train={len(train_idx)}\n"
                f"n_test={len(test_idx)}\n"
            )
        np.save(fold_dir / "train_idx.npy", train_idx)
        np.save(fold_dir / "test_idx.npy",  test_idx)

        results.append({
            "fold": i+1,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_elbo": train_elbo,
            "elbos": np.asarray(elbos),
            "latent_dim": latent_dim_i,
            "achieved_cum_variance": achieved,
        })

    return results

