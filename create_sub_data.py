# src/create_sub_data.py
import h5py, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List

def _macro_area(area: str) -> str | None:
    a = area.lower()
    if a.startswith("d"): return "dorsal"
    if a.startswith("v"): return "ventral"
    if a == "thalamus":   return "thalamus"
    return None

def export_area_splits(
    base_dir  : str | Path,
    rat_id    : int | str,
    units_csv : str | Path,
    *,
    h5_prefix : str = "NpxFiringRate",
    out_dir   : str | Path = None,
    min_spikes: int = 1,
    verbose   : bool = True
) -> Dict[str, List[str]]:
    base_dir = Path(base_dir)
    rat_tag  = f"Rat{int(rat_id)}"
    rat_dir  = base_dir / rat_tag
    h5_path  = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))

    if out_dir is None:
        out_dir = rat_dir / f"{rat_tag}_area_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        raise ValueError(f"No matching rows for {rat_tag}")

    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f["speed"][...]
        rates_all = f["firing_rates"][...]
        spike_grp = f["spike_times"]
        raw_keys  = list(spike_grp.keys())

    # direct mapping
    key2idx = {k: i for i, k in enumerate(raw_keys)}
    counts  = np.array([ spike_grp[k].size for k in raw_keys ])
    active  = counts >= min_spikes

    # filter metadata
    units = units[units["hdf5_key"].isin(raw_keys)].copy()
    units["row_idx"] = units["hdf5_key"].map(key2idx).astype(int)
    units = units[ active[units["row_idx"]] ]

    if verbose:
        print(f"{rat_tag}: active units ≥{min_spikes} spikes → {len(units)}")

    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units[units["macro"] == macro]
        if sub.empty: continue
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
            recs.extend(
                dict(time_s=t, speed=s,
                     cluster=row.cluster, neuron_type=row.neuron_type,
                     area=row.area, macro_area=macro, rate=r)
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


