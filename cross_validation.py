def cross_validate_rslds(
    fit_fn,               # your model-fitting function (should accept FR_z, u, ...)
    FR_z,                 # z-scored firing rates (T, N)
    u,                    # exogenous inputs (T, M)
    k=5,
    fit_kwargs=None,      # dict of any extra kwargs for the fit function
    verbose=True
):
    """Run k-fold cross-validation, return list of train/test logliks."""
    fit_kwargs = fit_kwargs or {}
    splits = contiguous_kfold_indices(len(FR_z), k)
    results = []
    for i, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"CV fold {i+1}/{k} ({len(train_idx)} train, {len(test_idx)} test)")
        res = fit_fn(
            FR_z=FR_z[train_idx], u=u[train_idx], **fit_kwargs
        )
        model = res["model"]
        # Compute held-out log-likelihood (or ELBO) on test
        ll_train = model.log_likelihood(FR_z[train_idx], inputs=u[train_idx])
        ll_test  = model.log_likelihood(FR_z[test_idx],  inputs=u[test_idx])
        results.append(dict(
            fold=i,
            train_loglik=ll_train,
            test_loglik=ll_test,
            elbos=res.get("elbos", None),
            model=model
        ))
        if verbose:
            print(f"  Train: {ll_train:.1f}  Test: {ll_test:.1f}")
    return results
