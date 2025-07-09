 # rSLDS/create_sub_data.py
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

def _macro_area(area: str) -> str | None:
    """Map fine labels to coarse macro-areas."""
    a = area.lower()
    if a.startswith("d"): return "dorsal"
    if a.startswith("v"): return "ventral"
    if a == "thalamus":   return "thalamus"
    return None

def export_area_splits(
    hdf5_path   : str | Path,
    units_csv   : str | Path,
    out_dir     : str | Path,
    rat_tag     : str,
    min_spikes  : int = 1,
    verbose     : bool = True
) -> Dict[str, List[str]]:
    """
    Create dorsal/ventral/thalamus splits from one rat's HDF5 + units.csv.
    hdf5_path must be the .hdf5 file itself, not its folder.
    rat_tag    a string like "Rat3" or "Rat15".
    """
    # prepare output folder
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read metadata
    units = (
        pd.read_csv(units_csv)
          .query("rat == @rat_tag")
          .assign(
             hdf5_key=lambda df: "cluster" + df["cluster"].astype(str),
             macro    = lambda df: df["area"].map(_macro_area)
          )
          .dropna(subset=["macro"])
    )
    if units.empty:
        raise ValueError(f"No rows for {rat_tag} in {units_csv}")

    # open HDF5 and pull everything we need *inside* the with
    with h5py.File(hdf5_path, "r") as f:
        time_vec   = f["time"][...]
        speed      = f["speed"][...]
        rates_all  = f["firing_rates"][...]      # (n_units × T)
        spike_grp  = f["spike_times"]
        raw_keys   = list(spike_grp.keys())      # e.g. cluster1234_0

        # build clean-to-raw mapping and count spikes *while file is open*
        clean_keys   = [k.split("_")[0] for k in raw_keys]
        clean2raw    = dict(zip(clean_keys, raw_keys))
        key2idx      = {clean: i for i, clean in enumerate(clean_keys)}
        spike_counts = np.array([spike_grp[raw].size
                                 for raw in raw_keys])
        is_active    = spike_counts >= min_spikes

    # now file is closed, but we have time_vec, speed, rates_all,
    # clean2raw, key2idx, is_active, and units DataFrame

    # keep only units present in HDF5 and active
    units = (
        units[units["hdf5_key"].isin(clean2raw)]
             .assign(row_idx=lambda df: df["hdf5_key"].map(key2idx))
    )
    units = units[units["row_idx"].notna() & is_active[units["row_idx"].astype(int)]]
    units["row_idx"] = units["row_idx"].astype(int)

    if verbose:
        print(f"{rat_tag}: active units ≥{min_spikes} spikes → {len(units)}")

    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units[units["macro"] == macro]
        if sub.empty:
            if verbose: print(f"[{macro}] skipped (no units)")
            continue
        if verbose: print(f"[{macro}] {len(sub)} neurons")

        # wide
        wide = pd.DataFrame(rates_all[sub.row_idx].T,
                            columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)

        # long
        recs = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            recs.extend({
                "time_s"     : t,
                "speed"      : s,
                "cluster"    : row.cluster,
                "neuron_type": row.neuron_type,
                "area"       : row.area,
                "macro_area" : macro,
                "rate"       : r
            } for t, s, r in zip(time_vec, speed, rates))

        long_path = out_dir / f"{macro}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)

        exported[macro] = [str(wide_path), str(long_path)]
        if verbose:
            print(f"[{macro}] wrote {wide_path.name} & {long_path.name}")

    if verbose:
        print(f"✓ All splits saved in: {out_dir}/")

    return exported
