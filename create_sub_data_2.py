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
    Return {resp_type: dict(time, speed, rates, metadata)} for the given rat.

    * rates shape: (T, N_resp) — time first, neurons second
    * metadata rows match those columns
    """
    # 1) match metadata to HDF5 indices
    mapping = match_units_to_hdf5(
        units_csv, base_dir, rat_id,
        h5_prefix=h5_prefix, verbose=False
    )

    # 2) load everything you need from HDF5, _inside_ the with‐block
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec   = f["time"][...]
        speed      = f["speed"][...] if "speed" in f else None
        # transpose so we get (T, N_total)
        rates_all  = f["firing_rates"][...].T

        # pull out the raw cluster keys and counts
        raw_keys     = list(f[spike_group].keys())  
        spike_counts = np.array([
            f[spike_group][k].size
            for k in raw_keys
        ])

    # 3) activity mask
    active_mask = spike_counts >= min_spikes
    keep_idx    = mapping["firing_rate_index"].values
    mapping     = mapping[active_mask[keep_idx]]

    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} units ≥ {min_spikes} spikes")

    # 4) split out each responsiveness type
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

