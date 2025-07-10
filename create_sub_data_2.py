# src/create_sub_data_2.py
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from match_data_with_metadata import match_units_to_hdf5
from load_dataset                 import load_rat_data

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
    Return a dict mapping each response‐type (from the 'shocks_response' column)
    to a sub‐dataset dict with keys ['time','speed','rates','metadata'].

    - rates: (T, N_resp)
    - metadata: subset of the units.csv rows matching those neurons
    """
    # 1) metadata ↔ HDF5 index
    mapping = match_units_to_hdf5(
        units_csv, base_dir, rat_id,
        h5_prefix=h5_prefix, verbose=False
    )
    # now `mapping` has all columns from units.csv, including 'shocks_response'

    # 2) pull raw data out of HDF5 in one block
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_file = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_file, "r") as f:
        time_vec  = f["time"][...]
        speed     = f["speed"][...] if "speed" in f else None
        rates_all = f["firing_rates"][...].T    # (T, n_units)
        spikes    = f[spike_group]              # group of spike‐time arrays
        # copy spike‐counts into a plain array
        raw_keys      = list(spikes.keys())
        spike_counts  = np.array([ spikes[k].size for k in raw_keys ])
        active_mask   = spike_counts >= min_spikes
        # build map: raw_keys → firing_rate_index
        key2idx       = {k: i for i,k in enumerate(raw_keys)}

    # 3) restrict mapping to only those with ≥min_spikes
    idxs = mapping.firing_rate_index.values
    keep = active_mask[idxs]
    mapping = mapping.loc[keep].reset_index(drop=True)
    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} units ≥ {min_spikes} spikes")

    # 4) split out by the *shocks_response* column
    out = {}
    for resp in responsive_types:
        sub = mapping[mapping["shocks_response"] == resp]
        if sub.empty:
            if verbose:
                print(f"{resp:>12s}: none → skipped")
            continue
        idx = sub.firing_rate_index.values
        out[resp] = dict(
            time     = time_vec,
            speed    = speed,
            rates    = rates_all[:, idx],
            metadata = sub.copy()
        )
        if verbose:
            print(f"{resp:>12s}: {len(idx)} units")

    return out
