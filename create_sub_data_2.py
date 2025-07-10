# rSLDS/create_sub_data_2.py
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from match_data_with_metadata import match_units_to_hdf5
from load_dataset            import load_rat_data

__all__ = ["get_responsive_subsets"]

def get_responsive_subsets(
    base_dir,
    rat_id,
    *,
    units_csv,
    h5_prefix="NpxFiringRate",
    spike_group="spike_times",
    min_spikes=1,
    response_types=("excited", "inhibited", "unresponsive"),
    verbose=True
):
    """
    Return a dict mapping each shocks_response → sub‐dataset for that rat.
    
    Each value is a dict with keys:
      - time    : (T,) array
      - speed   : (T,) array or None
      - rates   : (T, N_resp) firing‐rate matrix
      - metadata: DataFrame of the matching rows from units.csv

    Parameters
    ----------
    base_dir       : str/Path to folder containing RatXX/
    rat_id         : which Rat number (e.g. 10 → “Rat10”)
    units_csv      : master units.csv (with a “shocks_response” column)
    min_spikes     : drop units with fewer than this many spikes total
    response_types : tuple of strings to keep (case‐insensitive)
    """
    # 1) get metadata ↔ hdf5 index mapping
    mapping = match_units_to_hdf5(
        units_csv, base_dir, rat_id,
        h5_prefix=h5_prefix, verbose=False
    )
    # make response column lowercase for matching
    mapping["shocks_response"] = mapping["shocks_response"].str.lower()

    # 2) load HDF5 fields
    rat_dir = Path(base_dir)/f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f["speed"][...] if "speed" in f else None
        rates_all = f["firing_rates"][...].T   # (T, total_neurons)
        spikes    = f[spike_group]

    # 3) drop low‐spiking units
    spike_counts = np.array([spikes[k].size for k in spikes])
    active       = spike_counts >= min_spikes
    keep_idx     = mapping.firing_rate_index.values
    mapping      = mapping.loc[active[keep_idx]].reset_index(drop=True)
    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} units ≥ {min_spikes} spikes")

    # 4) build subsets per response_type
    out = {}
    for resp in response_types:
        key = resp.lower()
        sub = mapping[mapping["shocks_response"] == key]
        if sub.empty:
            if verbose:
                print(f"  → no “{resp}” units found, skipping")
            continue
        idx = sub.firing_rate_index.values
        out[key] = {
            "time":     time_vec,
            "speed":    speed,
            "rates":    rates_all[:, idx],
            "metadata": sub.copy()
        }
        if verbose:
            print(f"  • {resp:<12s}: {len(idx)} units")
    return out
