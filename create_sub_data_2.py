# src/create_sub_data_2.py
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from match_data_with_metadata import match_units_to_hdf5
from load_dataset               import load_rat_data

__all__ = ["get_responsive_subsets"]

def get_responsive_subsets(
    base_dir,
    rat_id,
    *,
    units_csv,
    h5_prefix       = "NpxFiringRate",
    spike_group     = "spike_times",
    min_spikes      = 1,
    responsive_types= ("excited", "inhibited"),
    verbose         = True
):
    """
    Return a dict of {resp_type: {time, speed, rates, metadata}} for one rat.

    - time     : (T,)         array of time points
    - speed    : (T,) or None running speed if present
    - rates    : (T, N_resp)  firing‐rates for each responsive neuron
    - metadata : pd.DataFrame  subset of units.csv matching those neurons
    """
    # 1) match metadata rows to HDF5 spike‐indices
    mapping = match_units_to_hdf5(
        units_csv, base_dir, rat_id,
        h5_prefix=h5_prefix, verbose=False
    )

    # 2) load HDF5 once
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f["speed"][...] if "speed" in f else None
        # firing_rates in file are stored as (neurons × time), we want (time × neurons)
        rates_all = f["firing_rates"][...].T
        spikes    = f[spike_group]

    # 3) activity mask (≥ min_spikes per neuron)
    spike_counts = np.array([spikes[k].size for k in spikes.keys()])
    active_mask  = spike_counts >= min_spikes
    keep_idx     = mapping["firing_rate_index"].values
    mapping      = mapping[active_mask[keep_idx]]

    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} units ≥ {min_spikes} spikes")

    # 4) split by responsiveness
    out = {}
    for resp in responsive_types:
        sub = mapping[mapping["neuron_type"] == resp]
        if sub.empty:
            if verbose:
                print(f"{resp:>9s}: none found → skipped")
            continue

        idx = sub["firing_rate_index"].values
        out[resp] = {
            "time":     time_vec,
            "speed":    speed,
            "rates":    rates_all[:, idx],
            "metadata": sub.reset_index(drop=True)
        }
        if verbose:
            print(f"{resp:>9s}: {len(idx)} units → subset built")

    return out
