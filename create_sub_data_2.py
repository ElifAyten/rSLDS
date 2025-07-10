import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from match_data_with_metadata import match_units_to_hdf5   # match CSV metadata to HDF5 indices
from load_dataset  import load_rat_data          # load time, speed, etc.

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
    Return a dict of sub-datasets for each response type *and* a combined "responsive" key.

    Keys:
      - each entry in `responsive_types` → dict with keys: time, speed, rates, metadata
      - "responsive" → dict with both excited+inhibited concatenated

    Rates shape: (T, N_cells), metadata rows match columns order.
    """
    # 1) metadata ↔ spike_index mapping
    mapping = match_units_to_hdf5(
        units_csv, base_dir, rat_id,
        h5_prefix=h5_prefix, verbose=False
    )

    # 2) load from HDF5
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f.get("speed")[...] if "speed" in f else None
        rates_all = f["firing_rates"][...].T   # (T, total_neurons)
        spikes    = f[spike_group]

    # 3) apply activity mask
    spike_counts = np.array([spikes[k].size for k in spikes.keys()])
    active       = spike_counts >= min_spikes
    idx_all      = mapping["firing_rate_index"].values
    mapping      = mapping[active[idx_all]].reset_index(drop=True)

    if verbose:
        print(f"Rat{rat_id}: {len(mapping)} active units ≥ {min_spikes} spikes")

    # 4) build per-type subsets
    out = {}
    for resp in responsive_types:
        sub = mapping[mapping["neuron_type"] == resp]
        if sub.empty:
            if verbose:
                print(f"{resp}: none → skipped")
            continue
        idx = sub["firing_rate_index"].values
        out[resp] = {
            "time":     time_vec,
            "speed":    speed,
            "rates":    rates_all[:, idx],
            "metadata": sub.copy()
        }
        if verbose:
            print(f"{resp}: {len(idx)} cells")

    # 5) combine excited + inhibited into "responsive"
    combined_keys = [k for k in responsive_types if k in out]
    if combined_keys:
        # concat along neuron axis
        rates_list = [out[k]["rates"] for k in combined_keys]
        meta_list  = [out[k]["metadata"] for k in combined_keys]
        rates_resp = np.concatenate(rates_list, axis=1)
        meta_resp  = pd.concat(meta_list, ignore_index=True)
        out["responsive"] = {
            "time":     time_vec,
            "speed":    speed,
            "rates":    rates_resp,
            "metadata": meta_resp
        }
        if verbose:
            total = rates_resp.shape[1]
            print(f"responsive: combined {total} cells ({'+'.join(combined_keys)})")

    return out
