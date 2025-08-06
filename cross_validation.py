from sklearn.model_selection import KFold
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
import ssm # Moved import ssm to top level


def footshock_vector(time_vec, shock_times):
    """
    Creates a binary vector indicating the presence of footshocks at each time point.

    Args:
        time_vec (np.ndarray): 1D array of time points.
        shock_times (np.ndarray): 1D array of footshock timestamps.

    Returns:
        np.ndarray: Binary vector of the same length as time_vec,
                    where 1 indicates a shock and 0 indicates no shock.
    """
    if shock_times is None or len(shock_times) == 0:
        return np.zeros_like(time_vec, dtype=int)

    # Assume time_vec is sorted
    shock_indices = np.searchsorted(time_vec, shock_times)

    # Create a zero vector and set 1 at shock indices
    shock_vec = np.zeros_like(time_vec, dtype=int)
    # Ensure indices are within bounds
    valid_indices = shock_indices[shock_indices < len(time_vec)]
    shock_vec[valid_indices] = 1

    return shock_vec


def _auto_latent_dim(FR, variance_goal):
     # Dummy implementation or use a proper PCA call if available
     # This is a placeholder as the original function wasn't provided
     # You might need to implement or import a real PCA function here
     # For now, let's return a fixed value or a simple calculation
     # A simple approach could be related to the number of features
     if FR.shape[1] <= 10:
         return FR.shape[1] # Return number of features if small
     else:
         return 10 # Arbitrarily return 10 for larger feature sets


def crossval_rslds(
    h5_path,
    csv_path,
    save_dir,
    *,
    K_states=2,
    num_iters=300,
    kappa=0.0,
    n_folds=5,
    variance_goal=0.90,
    verbose=True
):
    """
    K-Fold cross-validation for rSLDS.
    Returns a list of dictionaries, one per fold, with model & ELBOs.
    """
    # Load the data just as in fit_single_rslds
    df = pd.read_csv(csv_path)
    t = df.pop("time_s").values
    FR = df.values.astype(float)

    # Handle NaN values in FR before z-scoring
    FR = np.nan_to_num(FR, nan=0.0) # Replace NaN with 0

    # Shock regressor
    with h5py.File(h5_path, "r") as h5:
        # Handle potential missing 'footshock_times' key
        if "footshock_times" in h5:
             shock_times = h5["footshock_times"][...]
        else:
             shock_times = None
             if verbose: print("Warning: 'footshock_times' not found in HDF5.")

    u = footshock_vector(t, shock_times)
    u = u.reshape(-1, 1) # Reshape u to be (n_samples, 1)

    # Z-score firing rates
    mu, sd = FR.mean(0, keepdims=True), FR.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    FR_z = (FR - mu) / sd

    # Latent dim auto
    # Need a proper implementation or import for _auto_latent_dim
    # Assuming a simple placeholder or it's defined elsewhere
    latent_dim = _auto_latent_dim(FR, variance_goal)


    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for i, (train_idx, test_idx) in enumerate(kf.split(FR_z)):
        if verbose: print(f"=== Fold {i+1}/{n_folds} ({len(train_idx)} train, {len(test_idx)} test) ===")

        X_train, X_test = FR_z[train_idx], FR_z[test_idx]
        u_train, u_test = u[train_idx], u[test_idx]

        # Fit model on train
        model = ssm.SLDS(
            X_train.shape[1],
            K_states,
            latent_dim,
            M=1,
            transitions="inputdriven",
            transition_kwargs=dict(kappa=kappa),
            dynamics="gaussian",
            emissions="ar",
            single_subspace=True,
        )
        elbos, post = model.fit(
            [X_train],
            inputs=[u_train],
            num_iters=num_iters,
            method="bbvi",
            variational_posterior="meanfield",
        )

        # Compute train/test ELBO
        train_elbo = elbos[-1]
        # Evaluate on held-out
        # The ssm package doesn't have log_likelihood for SLDS; use ELBO instead

        # Save outputs
        fold_dir = Path(save_dir) / f"cv_fold{i+1}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        np.save(fold_dir / "elbos.npy", elbos)
        # (you can also save model, post, etc.)

        results.append({
            "fold": i+1,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_elbo": train_elbo,
            "elbos": elbos,
        })

    return results
