import numpy as np

def contiguous_kfold_indices(T, k=5):
    """Return list of (train_idx, test_idx) for contiguous k-fold splits."""
    indices = np.arange(T)
    fold_sizes = np.full(k, T // k)
    fold_sizes[:T % k] += 1
    splits = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = np.arange(start, stop)
        train_idx = np.concatenate([np.arange(0, start), np.arange(stop, T)])
        splits.append((train_idx, test_idx))
        current = stop
    return splits
