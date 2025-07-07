__all__ = ["normalize_firing_rates"]

def normalize_firing_rates(fr, *, time_first=True, return_scaler=False):
    """
    Z-score each neuron's firing-rate vector independently.

    Parameters
    ----------
    fr : ndarray
        Firing-rate matrix. Shape (T, N) if `time_first=True`
        else (N, T).
    time_first : bool, default True
        True  -> rows=time, cols=neurons  (common for numpy/ML)
        False -> rows=neurons, cols=time  (original ephys style)
    return_scaler : bool, default False
        If True, also return the fitted sklearn StandardScaler.

    Returns
    -------
    fr_z : ndarray
        Normalized firing rates, same shape as input.
    scaler : StandardScaler  (only if return_scaler=True)
    """
    X = fr if time_first else fr.T          # ensure (samples, features)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    fr_z = Xz if time_first else Xz.T
    if return_scaler:
        return fr_z, scaler
    return fr_z

    
