# src/create_sub_data_2.py
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from metadata import match_units_to_hdf5   # now found, because src/ is on sys.path
from dataset  import load_rat_data          # ditto

__all__ = ["get_responsive_subsets"]

def get_responsive_subsets(
    base_dir,
    rat_id,
    *,
    units_csv,
    h5_prefix="NpxFiringRate",
    spike_group="spike_times",
    min_spikes=1,
    responsive_types=("excited", "inhibited"),
    verbose=True
):
    """
    Return {resp_type: dict(time, speed, rates, metadata)} for the given rat.

    * rates shape: (T, N_resp)  — time first, neurons second
    * metadata rows match those columns
    """
    # 1) metadata ↔ spike_index
    mapping = match_units_to_hdf5(
        units_csv, base_dir, rat_id,
        h5_prefix=h5_prefix, verbose=False
    )

    # 2) load everything from HDF5
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f.get("speed", None)[...] if "speed" in f else None
        rates_all = f["firing_rates"][...].T   # shape (T, N_total)
        spikes    = f[spike_group]

    # 3) activity mask
    spike_counts = np.array([spikes[k].size for k in spikes])
    active       = spike_counts >= min_spikes
    keep_idx     = mapping.firing_rate_index.values
    mapping      = mapping[active[keep_idx]]

    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} units ≥ {min_spikes} spikes")

    out = {}
    for resp_type in responsive_types:
        sub = mapping[mapping.neuron_type == resp_type]
        if sub.empty:
            if verbose:
                print(f"{resp_type}: none ➞ skipped")
            continue
        idx = sub.firing_rate_index.values
        out[resp_type] = dict(
            time     = time_vec,
            speed    = speed,
            rates    = rates_all[:, idx],
            metadata = sub.reset_index(drop=True)
        )
        if verbose:
            print(f"{resp_type:>9s}: {len(idx)} units")
    return out
