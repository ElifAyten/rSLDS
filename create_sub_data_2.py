import h5py, numpy as np, pandas as pd
from pathlib import Path
from metadata import match_units_to_hdf5          # ← your existing helper
from dataset   import load_rat_data               # ← loads speed, etc.

__all__ = ["get_responsive_subsets"]

def get_responsive_subsets(base_dir,
                           rat_id,
                           *,
                           units_csv,
                           h5_prefix="NpxFiringRate",
                           spike_group="spike_times",
                           min_spikes=1,
                           responsive_types=("excited", "inhibited"),
                           verbose=True):
    """
    Return {resp_type: dict(time, speed, rates, metadata)} for the given rat.

    * rates shape: (T, N_resp) — NOT transposed (time first, neurons second)
    * metadata rows are a subset of units.csv that match those columns.
    """

    # 1. match metadata ↔ HDF5 spike indices
    mapping = match_units_to_hdf5(units_csv, base_dir, rat_id,
                                  h5_prefix=h5_prefix, verbose=False)

    # 2. load HDF5 once
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec = f["time"][...]
        speed    = f["speed"][...] if "speed" in f else None
        rates_all= f["firing_rates"][...].T       # (T, total_neurons)
        spikes   = f[spike_group]

    # 3. activity mask
    spike_counts = np.array([spikes[k].size for k in spikes.keys()])
    active = spike_counts >= min_spikes
    mapping = mapping[active[mapping.firing_rate_index.values]]

    if verbose:
        print(f"Rat{rat_id}: active units ≥{min_spikes} spikes → {len(mapping)}")

    out = {}
    for t in responsive_types:
        sub = mapping[mapping.neuron_type == t]
        if sub.empty:
            if verbose:
                print(f"{t}: none found → skipped")
            continue

        idx = sub.firing_rate_index.values
        out[t] = dict(time=time_vec,
                      speed=speed,
                      rates=rates_all[:, idx],   # (T, N_resp)
                      metadata=sub.reset_index(drop=True))

        if verbose:
            print(f"{t:>9s}: {len(idx):3d} units  → sub-dataset built")

    return out
