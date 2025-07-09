# src/create_sub_data.py
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union

def _macro_area(area: str) -> str:
    """
    Map fine-grained area names to one of: 'dorsal', 'ventral', or 'thalamus'.
    Anything else returns None (and gets dropped).
    """
    a = area.lower()
    if a.startswith("d"):
        return "dorsal"
    if a.startswith("v"):
        return "ventral"
    if a == "thalamus":
        return "thalamus"
    return None

def export_area_splits(
    base_dir: Union[str, Path],
    rat_id: Union[int, str],
    units_csv: Union[str, Path],
    *,
    h5_prefix: str = "NpxFiringRate",
    out_dir: Union[str, Path] = None,
    min_spikes: int = 1,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Build wide- and long-format CSVs for each macro-area (dorsal / ventral /
    thalamus). Returns a dict: { macro_area: [wide_csv, long_csv], … }.

    Parameters
    ----------
    base_dir   : folder that contains subfolders RatXX/
    rat_id     : which rat number (e.g. 3 → "Rat3")
    units_csv  : path to master units.csv
    h5_prefix  : filename prefix for your .hdf5 files
    out_dir    : where to write the CSVs (default = RatXX/RatXX_area_splits/)
    min_spikes : only keep neurons with ≥ this many spikes
    verbose    : print progress
    """
    base_dir = Path(base_dir)
    rat_tag  = f"Rat{int(rat_id)}"
    rat_dir  = base_dir / rat_tag

    # find the HDF5 file
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))

    # choose output directory
    if out_dir is None:
        out_dir = rat_dir / f"{rat_tag}_area_splits"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load your units metadata
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

    # open HDF5 once
    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f["speed"][...]
        rates_all = f["firing_rates"][...]  # shape (n_units, T)
        spike_grp = f["spike_times"]
        raw_keys  = list(spike_grp.keys())  # e.g. ['cluster1031_0', ...]

    # build direct map raw_key → row index
    key2idx = {k: i for i, k in enumerate(raw_keys)}

    # spike-counts for filtering
    counts = np.array([ spike_grp[k].size for k in raw_keys ])
    active = counts >= min_spikes

    # filter your metadata to only those keys present & active
    units = units[units["hdf5_key"].isin(raw_keys)].copy()
    units["row_idx"] = units["hdf5_key"].map(key2idx).astype(int)
    units = units[ active[units["row_idx"]] ]

    if verbose:
        print(f"{rat_tag}: {len(units)} active units (≥{min_spikes} spikes)")

    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units[units["macro"] == macro]
        if sub.empty:
            continue
        if verbose:
            print(f"[{macro}] exporting {len(sub)} neurons")

        # wide‐format CSV
        wide = pd.DataFrame(rates_all[sub.row_idx].T,
                            columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)

        # long‐format (tidy) CSV
        recs = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            recs.extend(
                {
                    "time_s"    : t,
                    "speed"     : s,
                    "cluster"   : row.cluster,
                    "neuron_type": row.neuron_type,
                    "area"      : row.area,
                    "macro_area": macro,
                    "rate"      : r
                }
                for t, s, r in zip(time_vec, speed, rates)
            )
        long_path = out_dir / f"{macro}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)

        exported[macro] = [str(wide_path), str(long_path)]
        if verbose:
            print(f"  → wrote {wide_path.name}, {long_path.name}")

    if verbose:
        print(f"✓ CSVs saved in: {out_dir}/")

    return exported

