import pandas as pd
import numpy as np
import tempfile
import shutil

def crossval_rslds_csv(
    h5_path,
    csv_path,
    save_dir,
    fit_fn,
    K_states=2,
    num_iters=300,
    kappa=0,
    k=5,
    overwrite=True,
    verbose=True,
):
    """Run k-fold CV for rSLDS using file-based split."""
    df = pd.read_csv(csv_path)
    T = len(df)
    fold_sizes = np.full(k, T // k)
    fold_sizes[:T % k] += 1
    indices = np.arange(T)
    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = np.arange(start, stop)
        train_idx = np.concatenate([np.arange(0, start), np.arange(stop, T)])
        splits.append((train_idx, test_idx))
        current = stop

    results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"\n=== Fold {i+1}/{k} ({len(train_idx)} train, {len(test_idx)} test) ===")
        # Make temp dirs/files for train/test
        with tempfile.TemporaryDirectory() as temp_dir:
            train_csv = f"{temp_dir}/train.csv"
            test_csv  = f"{temp_dir}/test.csv"
            df.iloc[train_idx].to_csv(train_csv, index=False)
            df.iloc[test_idx].to_csv(test_csv, index=False)
            
            # Each fold's model goes in a subfolder
            fold_save = save_dir / f"cv_fold_{i+1}"
            fold_save.mkdir(parents=True, exist_ok=True)
            
            # Fit model on train split
            out = fit_fn(
                h5_path   = h5_path,
                csv_path  = train_csv,
                save_dir  = fold_save,
                K_states  = K_states,
                num_iters = num_iters,
                kappa     = kappa,
                overwrite = overwrite,
                verbose   = False,
            )
            model = out["model"]
            
            # Compute train and test log-likelihoods
            import h5py
            FR_z_train = pd.read_csv(train_csv).drop(columns=["time_s", "speed"], errors='ignore').values
            FR_z_test  = pd.read_csv(test_csv).drop(columns=["time_s", "speed"], errors='ignore').values
            with h5py.File(h5_path, "r") as h5:
                speed_train = pd.read_csv(train_csv)["speed"].values if "speed" in df.columns else h5["speed"][...][train_idx]
                speed_test  = pd.read_csv(test_csv)["speed"].values  if "speed" in df.columns else h5["speed"][...][test_idx]
                u_train = speed_train.reshape(-1, 1)
                u_test  = speed_test.reshape(-1, 1)
            train_ll = model.log_likelihood(FR_z_train, inputs=u_train)
            test_ll  = model.log_likelihood(FR_z_test,  inputs=u_test)
            
            results.append({
                "fold": i+1,
                "train_loglik": train_ll,
                "test_loglik": test_ll,
                "save_dir": fold_save,
                "model": model,
            })
            if verbose:
                print(f"  Train LL: {train_ll:.2f}, Test LL: {test_ll:.2f}")
    return results

