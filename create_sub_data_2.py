# src/create_sub_data_2.py

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from metadata import match_units_to_hdf5   # your existing helper
from load_dataset  import load_rat_data          # your existing loader

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
) -> Dict[str, Dict]:
    """
    Return sub‐datasets keyed by response type AND a merged 'responsive' set.

    Args
    ----
    base_dir : str|Path
        Folder containing RatXX subfolders.
    rat_id : int|str
        Which rat (e.g. 3 → “Rat3”).
    units_csv : str|Path
        Master metadata file (units.csv).
    h5_prefix : str
        Filename prefix for your .hdf5 (default "NpxFiringRate").
    spike_group : str
        HDF5 group name for spike_times.
    min_spikes : int
        Minimum total spikes per cell to include.
    responsive_types : tuple[str]
        Which neuron_type values to pull; must match units.csv 'neuron_type'.
    Returns
    -------
    dict of dicts, keys = (*responsive_types, "responsive"):
      each value is a dict with keys:
        time     : (T,) array of time stamps
        speed    : (T,) or None
        rates    : (T, N_cells) firing‐rate matrix
        metadata : pandas.DataFrame subset of units.csv with extra cols
    """
    # 1. build full mapping (units.csv → HDF5 index)
    mapping = match_units_to_hdf5(units_csv, base_dir, rat_id,
                                  h5_prefix=h5_prefix, verbose=False)

    # 2. open HDF5 once
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_file = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_file, "r") as f:
        time_vec  = f["time"][...]
        speed     = f.get("speed", None)[...] if "speed" in f else None
        # firing_rates stored (neurons × time) → transpose → (T × total_neurons)
        rates_all = f["firing_rates"][...].T
        spikes    = f[spike_group]

    # 3. filter out low‐spiking cells
    counts = np.array([ spikes[k].size for k in spikes.keys() ])
    active = counts >= min_spikes
    idxs   = mapping["firing_rate_index"].values
    mapping = mapping.loc[active[idxs], :].reset_index(drop=True)

    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} cells with ≥{min_spikes} spikes")

    # 4. build per‐type subsets
    out: Dict[str, Dict] = {}
    for resp in responsive_types:
        sub = mapping[mapping["neuron_type"] == resp]
        if sub.empty:
            if verbose:
                print(f"{resp:>10s}: none → skipped")
            continue
        cols = sub["firing_rate_index"].values
        out[resp] = {
            "time":     time_vec,
            "speed":    speed,
            "rates":    rates_all[:, cols],
            "metadata": sub.copy()
        }
        if verbose:
            print(f"{resp:>10s}: {len(cols)} cells")

    # 5. merged “responsive” = all excited+inhibited
    merged_idx = []
    for resp in responsive_types:
        if resp in out:
            # collect their metadata indices
            merged_idx.extend(out[resp]["metadata"]["firing_rate_index"].tolist())
    if merged_idx:
        merged_idx = np.unique(merged_idx)
        merged_meta = mapping[mapping["firing_rate_index"].isin(merged_idx)].reset_index(drop=True)
        out["responsive"] = {
            "time":     time_vec,
            "speed":    speed,
            "rates":    rates_all[:, merged_idx],
            "metadata": merged_meta
        }
        if verbose:
            print(f"{'responsive':>10s}: {len(merged_idx)} cells (merged)")

    return out
