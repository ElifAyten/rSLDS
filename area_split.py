# src/area_split.py
import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def macro_area(area: str) -> str | None:
    """Map fine area names to coarse: dorsal, ventral, thalamus, or None."""
    a = area.lower()
    if a.startswith("d"):      return "dorsal"
    if a.startswith("v"):      return "ventral"
    if a == "thalamus":        return "thalamus"
    return None

def export_area_splits(
    hdf5_path  : str | Path,
    units_csv  : str | Path,
    out_dir    : str | Path,
    rat_tag    : str,
    min_spikes : int = 1
):
    """
    For one RatXX:
      - read its HDF5 and units.csv
      - select only clusters with ≥ min_spikes
      - split into dorsal/ventral/thalamus
      - write *_wide.csv and *_long.csv in out_dir
    Returns: list of written filepaths.
    """
    hdf5_path = Path(hdf5_path)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load metadata
    units = (
        pd.read_csv(units_csv)
          .query("rat == @rat_tag")
          .assign(hdf5_key = lambda df: "cluster" + df["cluster"].astype(str),
                  macro    = lambda df: df["area"].map(macro_area))
          .dropna(subset=["macro"])
    )
    # 2) load HDF5
    with h5py.File(hdf5_path, "r") as f:
        time_vec   = f["time"][...]
        speed      = f["speed"][...]
        rates_all  = f["firing_rates"][...]   # (n_units, T)
        spike_grp  = f["spike_times"]
        raw_keys   = list(spike_grp.keys())   # e.g. 'cluster1031_0'

    # direct key→row index
    key2row = {k: i for i, k in enumerate(raw_keys)}
    counts   = np.array([spike_grp[k].size for k in raw_keys])
    active   = counts >= min_spikes

    # keep only rows in units that appear in raw_keys & are active
    units = units[units["hdf5_key"].isin(raw_keys)].copy()
    units["row_idx"] = units["hdf5_key"].map(key2row)
    units = units[active[units["row_idx"].values]]

    out_files = []
    for macro in ["dorsal", "ventral", "thalamus"]:
        sub = units[units["macro"] == macro]
        if sub.empty:
            continue

        # wide
        wide = pd.DataFrame(rates_all[sub.row_idx].T,
                            columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)
        out_files.append(str(wide_path))

        # long
        records = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            for t, s, r in zip(time_vec, speed, rates):
                records.append({
                    "time_s"     : t,
                    "speed"      : s,
                    "cluster"    : row.cluster,
                    "neuron_type": row.neuron_type,
                    "area"       : row.area,
                    "macro_area" : macro,
                    "rate"       : r
                })
        long_df   = pd.DataFrame.from_records(records)
        long_path = out_dir / f"{macro}_long.csv"
        long_df.to_csv(long_path, index=False)
        out_files.append(str(long_path))

    return out_files


