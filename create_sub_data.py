# src/area_split.py
import h5py, numpy as np, pandas as pd
from pathlib import Path
from typing  import Dict, List


def _macro_area(area: str) -> str | None:
    """Map fine labels to coarse macro-areas."""
    a = area.lower()
    if a.startswith("d"): return "dorsal"
    if a.startswith("v"): return "ventral"
    if a == "thalamus":   return "thalamus"
    return None                           # anything else → drop


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
    """
    Build wide- and long-format CSVs for each macro-area (dorsal / ventral /
    thalamus).  Returns ``{macro_area: [wide_csv, long_csv], …}``.
    """
    #  locate I/O paths
    base_dir = Path(base_dir)
    rat_tag  = f"Rat{int(rat_id)}"
    rat_dir  = base_dir / rat_tag
    h5_path  = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))

    if out_dir is None:
        out_dir = rat_dir / f"{rat_tag}_area_splits"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # read metadata (units.csv)
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
        raise ValueError(f"No rows for {rat_tag} in {units_csv}")

    # open HDF5 once 
    with h5py.File(h5_path, "r") as f:
        time_vec  = f["time"][...]
        speed     = f["speed"][...]
        rates_all = f["firing_rates"][...]          # (n_units × T)
        spike_grp = f["spike_times"]
        raw_keys  = list(spike_grp.keys())          # ex: cluster1234_0

    #  build mapping clean-key → row-index & spike counts
    clean_keys  = [k.split("_")[0] for k in raw_keys]           # cluster1234
    clean2raw   = dict(zip(clean_keys, raw_keys))
    key2idx     = {clean: i for i, clean in enumerate(clean_keys)}
    spike_counts= np.array([spike_grp[clean2raw[c]].size for c in clean_keys])
    is_active   = spike_counts >= min_spikes

    # keep only units present in the file and active
    units = (
        units[units["hdf5_key"].isin(clean2raw)]
          .assign(row_idx=lambda d: d["hdf5_key"].map(key2idx).astype(int))
    )
    units = units[ is_active[units.row_idx.values] ]

    if verbose:
        print(f"{rat_tag}: active units ≥{min_spikes} spikes → {len(units)}")

    # export one pair of CSVs per macro-area
    exported: Dict[str, List[str]] = {}
    for macro in sorted(units["macro"].unique()):
        sub = units.loc[units["macro"] == macro]
        if sub.empty:
            continue
        if verbose:
            print(f"[{macro}] {len(sub)} neurons")

        # wide 
        wide = pd.DataFrame(rates_all[sub.row_idx].T,
                            columns=sub["cluster"].astype(str))
        wide.insert(0, "time_s", time_vec)
        wide["speed"] = speed
        wide_path = out_dir / f"{macro}_wide.csv"
        wide.to_csv(wide_path, index=False)

        #  long
        recs = []
        for _, row in sub.iterrows():
            rates = rates_all[row.row_idx]
            recs.extend(
                dict(time_s=t, speed=s, cluster=row.cluster,
                     neuron_type=row.neuron_type, area=row.area,
                     macro_area=macro, rate=r)
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

