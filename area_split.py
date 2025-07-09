# src/area_split.py
import os, h5py, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List

def macro_area(area: str) -> str | None:
    """Map fine area names to coarse: dorsal, ventral or thalamus."""
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
) -> List[str]:
    """
    For one RatXX:
      - read its HDF5 & units.csv
      - keep clusters with ≥min_spikes
      - split into dorsal/ventral/thalamus
      - write wide & long CSVs into out_dir
    Returns list of written filepaths.
    """
    hdf5_path = Path(hdf5_path)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) load unit metadata
    df = pd.read_csv(units_csv)
    units = (
        df.query("rat == @rat_tag")
          .assign(hdf5_key = lambda d: "cluster" + d["cluster"].astype(str),
                  macro    = lambda d: d["area"].map(macro_area))
          .dropna(subset=["macro"])
    )

    # --- 2) open HDF5 once
    with h5py.File(hdf5_path, "r") as f:
        time_vec  = f["time"][...]
        # gracefully handle missing 'speed'
        speed     = f["speed"][...] if "speed" in f else np.zeros_like(time_vec)
        rates_all = f["firing_rates"][...]    # shape (n_units, T)
        spike_grp = f["spike_times"]
        raw_keys  = list(spike_grp.keys())    # e.g. 'cluster1031_0'

    # build key→row index & spike‐counts mask
    key2row     = {k: i for i, k in enumerate(raw_keys)}
    counts      = np.array([spike_grp[k].size for k in raw_keys])
    is_active   = counts >= min_spikes

    # filter your units to just the ones present & active
    units = units[units["hdf5_key"].isin(raw_keys)].copy()
    units["row_idx"] = units["hdf5_key"].map(key2row)
    units = units[is_active[units["row_idx"].values]]

    out_files = []
    # for each macro‐area, build & save two CSVs
    for macro in ["dorsal", "ventral", "thalamus"]:
        sub = units[units["macro"] == macro]
        if sub.empty:
            continue

        # wide form
        wide = pd.DataFrame(
            rates_all[sub.row_idx].T,
            columns=sub["cluster"].astype(str)
        )
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)
        out_files.append(str(wide_path))

        # long (tidy) form
        recs = []
        for _, row in sub.iterrows():
            r_trace = rates_all[row.row_idx]
            for t, spd, r in zip(time_vec, speed, r_trace):
                recs.append({
                    "time_s"     : t,
                    "speed"      : spd,
                    "cluster"    : row.cluster,
                    "neuron_type": row.neuron_type,
                    "area"       : row.area,
                    "macro_area" : macro,
                    "rate"       : r
                })
        long_path = out_dir / f"{macro}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)
        out_files.append(str(long_path))

    return out_files
