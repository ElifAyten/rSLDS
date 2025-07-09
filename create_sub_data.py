# src/area_split.py
import os, h5py, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List

# ----------------------------------------------------------------------
def _macro_area(area: str) -> str | None:
    a = area.lower()
    if a.startswith("d"): return "dorsal"
    if a.startswith("v"): return "ventral"
    if a == "thalamus":   return "thalamus"
    return None

# ----------------------------------------------------------------------
def export_area_splits(
    base_dir      : str | Path,
    rat_id        : int | str,
    units_csv     : str | Path,
    *,
    h5_prefix     : str = "NpxFiringRate",
    out_dir       : str | Path = None,
    min_spikes    : int = 1,
    verbose       : bool = True
) -> Dict[str, List[str]]:
    """
    Create wide + long CSVs for each macro-area (dorsal / ventral / thalamus).

    Returns
    -------
    dict   { macro_area : [wide_csv_path, long_csv_path], … }
    """
    # ── locate files ────────────────────────────────────────────────────
    base_dir = Path(base_dir)
    rat_tag  = f"Rat{int(rat_id)}"
    rat_dir  = base_dir / rat_tag
    h5_path  = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))

    if out_dir is None:
        out_dir = rat_dir / f"{rat_tag}_area_splits"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── read unit metadata ─────────────────────────────────────────────
    units = (
        pd.read_csv(units_csv)
          .query("rat == @rat_tag")
          .assign(
              hdf5_key = lambda d: "cluster" + d["cluster"].astype(str),
              macro    = lambda d: d["area"].map(_macro_area)
          )
          .dropna(subset=["macro"])
    )
    if units.empty:
        raise ValueError(f"No matching rows for {rat_tag} in {units_csv}")

    # ── open HDF5 once ─────────────────────────────────────────────────
    with h5py.File(h5_path, "r") as f:
        time_vec   = f["time"][...]
        speed      = f["speed"][...]
        rates_all  = f["firing_rates"][...]          # (n_units × T)
        spike_grp  = f["spike_times"]
        raw_keys   = list(spike_grp.keys())          # e.g. cluster1234_0

    # ── map raw keys → cleaned (“cluster1234”) & row indices ───────────
    clean2raw   = {k.split("_")[0]: k for k in raw_keys}
    key2idx     = {clean: i for i, clean in enumerate(clean2raw)}
    spike_counts= np.array([spike_grp[k].size for k in raw_keys])
    is_active   = spike_counts >= min_spikes

    # keep only units that exist in HDF5 and are active
    units = (
        units[units["hdf5_key"].isin(clean2raw)]
          .assign(row_idx=lambda d: d["hdf5_key"].map(key2idx).astype(int))
    )
    units = units[is_active[units.row_idx.values]]

    if verbose:
        print(f"{rat_tag}: active units ≥{min_spikes} spikes → {len(units)}")

    # ── export per macro-area ──────────────────────────────────────────
    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units.query("macro == @macro")
        if sub.empty:
            continue
        if verbose:
            print(f"[{macro}] {len(sub)} neurons")

        # wide table ----------------------------------------------------
        wide = pd.DataFrame(rates_all[sub.row_idx].T,
                            columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)

        # long (tidy) table --------------------------------------------
        recs = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            recs.extend(
                {
                    "time_s"     : t,
                    "speed"      : s,
                    "cluster"    : row.cluster,
                    "neuron_type": row.neuron_type,
                    "area"       : row.area,
                    "macro_area" : macro,
                    "rate"       : r
                }
                for t, s, r in zip(time_vec, speed, rates)
            )
        long_path = out_dir / f"{macro}_long.csv"
        pd.DataFrame.from_records(recs).to_csv(long_path, index=False)

        exported[macro] = [str(wide_path), str(long_path)]
        if verbose:
            print(f"[{macro}] wrote {wide_path.name} & {long_path.name}")

    if verbose:
        print(f"✓ All splits saved in: {out_dir}/")
    return exported
