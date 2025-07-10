# src/create_sub_data_2.py
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from match_data_with_metadata import match_units_to_hdf5
from load_dataset             import load_rat_data

__all__ = ["get_responsive_subsets"]

def get_responsive_subsets(
    base_dir,
    rat_id,
    *,
    units_csv,
    h5_prefix="NpxFiringRate",
    spike_group="spike_times",
    min_spikes=1,
    responsive_types=("excited", "inhibited", "unresponsive"),
    verbose=True
):
    """
    Return {resp_type: dict(time, speed, rates, metadata)} for the given rat.

    rates shape: (T, N_resp) — time first, neurons second
    metadata rows match those columns.
    """
    # 1) metadata ↔ HDF5 index
    mapping = match_units_to_hdf5(units_csv, base_dir, rat_id,
                                  h5_prefix=h5_prefix, verbose=False)

    # 2) load EVERYTHING you need from the HDF5 while it's open
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_file = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_file, "r") as f:
        time_vec   = f["time"][...]
        speed      = f["speed"][...] if "speed" in f else None
        # (n_units × T) → transpose → (T, n_units)
        rates_all  = f["firing_rates"][...].T
        # pull out all spike‐time arrays into a dict so it stays alive
        spikes_dict = {k: f[spike_group][k][...] for k in f[spike_group].keys()}

    # 3) activity mask
    # now spikes_dict is just a python dict of numpy arrays
    spike_counts = np.array([len(v) for v in spikes_dict.values()])
    active_mask  = spike_counts >= min_spikes

    # only keep those rows in mapping whose firing_rate_index is active
    mapping = mapping.loc[active_mask[mapping.firing_rate_index.values]]
    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} units ≥ {min_spikes} spikes")

    # 4) split out each response type
    out = {}
    for resp in responsive_types:
        sub = mapping[mapping.neuron_type == resp]
        if sub.empty:
            if verbose:
                print(f"{resp}: none → skipped")
            continue
        idx = sub.firing_rate_index.values
        out[resp] = dict(
            time     = time_vec,
            speed    = speed,
            rates    = rates_all[:, idx],
            metadata = sub.reset_index(drop=True),
        )
        if verbose:
            print(f"{resp:>12s}: {len(idx)} units")

    return out
