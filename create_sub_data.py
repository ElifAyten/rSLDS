# src/create_sub_data.py
import os
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd


def _macro_area(area: str) -> str | None:
    """Map detailed area → coarse macro-area."""
    a = area.lower()
    if a.startswith("d"):   return "dorsal"
    if a.startswith("v"):   return "ventral"
    if a == "thalamus":     return "thalamus"
    return None


def export_area_splits(
    hdf5_path : str | Path,
    units_csv : str | Path,
    out_dir   : str | Path,
    rat_tag   : str,
    *,
    min_spikes: int   = 1,
    verbose   : bool  = True
) -> Dict[str, List[str]]:
    """
    For one RatXX HDF5 + the master units.csv, exports for each
    macro-area (dorsal/ventral/thalamus) two CSVs:
      - {out_dir}/{macro}_wide.csv
      - {out_dir}/{macro}_long.csv

    Returns a dict: { macro: [wide_csv, long_csv], … }.
    """
    hdf5_path = Path(hdf5_path)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load metadata & keep only this rat + areas that map to our macros
    units = (
        pd.read_csv(units_csv)
          .query("rat == @rat_tag")
          .assign(
              hdf5_key = lambda df: "cluster" + df["cluster"].astype(str),
              macro    = lambda df: df["area"].map(_macro_area)
          )
          .dropna(subset=["macro"])
    )
    if units.empty:
        raise ValueError(f"No rows for {rat_tag} in {units_csv}")

    # 2) open HDF5 once, grab the raw spike‐keys
    with h5py.File(hdf5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f.get("speed", None)
        rates_all = f["firing_rates"][...]     # shape = (n_units, T)
        spike_grp = f["spike_times"]
        raw_keys  = list(spike_grp.keys())     # e.g. "cluster1031_0"

    # 3) build index & spike counts
    key2row      = {k: i for i, k in enumerate(raw_keys)}
    spike_counts = np.array([ spike_grp[k].size for k in raw_keys ])
    is_active    = spike_counts >= min_spikes

    # 4) filter metadata down to exactly those keys & apply min_spikes
    units = units[ units["hdf5_key"].isin(raw_keys) ].copy()
    units["row_idx"] = units["hdf5_key"].map(key2row)
    units = units.iloc[ np.where(is_active[ units["row_idx"] ])[0] ]

    if verbose:
        print(f"{rat_tag}: {len(units)} units with ≥{min_spikes} spikes")

    exported: Dict[str, List[str]] = {}
    # 5) for each macro-area, dump wide + long CSVs
    for macro in sorted(units["macro"].unique()):
        sub = units[ units["macro"] == macro ]
        if sub.empty:
            continue
        if verbose:
            print(f"[{macro}] exporting {len(sub)} neurons…")

        # — wide format —
        wide = pd.DataFrame(
            rates_all[sub["row_idx"], :].T,
            columns=sub["cluster"].astype(str)
        )
        wide.insert(0, "time_s", time_vec)
        if speed is not None:
            wide["speed"] = speed
        wide_csv = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_csv, index=False)

        # — long format —
        recs = []
        for _, row in sub.iterrows():
            r_idx = int(row["row_idx"])
            for t, r in zip(time_vec, rates_all[r_idx]):
                recs.append({
                    "time_s"    : t,
                    "cluster"   : row["cluster"],
                    "area"      : row["area"],
                    "macro_area": macro,
                    "rate"      : r,
                    **({"speed": float(speed[np.searchsorted(time_vec, t)])}
                       if speed is not None else {})
                })
        long = pd.DataFrame.from_records(recs)
        long_csv = out_dir / f"{macro}_long.csv"
        long.to_csv(long_csv, index=False)

        exported[macro] = [str(wide_csv), str(long_csv)]
        if verbose:
            print(f"  → wrote {wide_csv.name}, {long_csv.name}")

    if verbose:
        print(f"✔️  All splits saved in: {out_dir}")
    return exported
