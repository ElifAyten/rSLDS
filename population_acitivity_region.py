
import numpy as np
import pandas as pd
import h5py, matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from .match_data_with_metadata import match_units_to_hdf5  


__all__ = ["population_activity_by_region"]

def population_activity_by_region(base_dir,
                                  rat_id,
                                  *,
                                  units_csv,
                                  win_ms       = 10,
                                  h5_prefix    = "NpxFiringRate",
                                  spike_group  = "spike_times",
                                  do_plot      = False,
                                  verbose      = True):
    """
    Histogram spikes for every neuron, then sum & average per brain area.

    Returns
    -------
    dict
      keys = area names
      each value = {"time": 1-D array, "total": counts, "mean": mean_per_neuron,
                    "n_neurons": int}
    """

    # match metadata rows to spike indices 
    mapping = match_units_to_hdf5(units_csv, base_dir, rat_id,
                                  h5_prefix=h5_prefix, verbose=False)

    # open HDF5 
    rat_dir = Path(base_dir) / f"Rat{int(rat_id)}"
    h5_path = next(rat_dir.glob(f"{h5_prefix}*.hdf5"))
    with h5py.File(h5_path, "r") as f:
        time_vec        = f["time"][...]
        footshock_times = f["footshock_times"][...]
        spikes_grp      = f[spike_group]

        dt        = time_vec[1] - time_vec[0]            # seconds
        desired_dt= win_ms / 1000
        if abs(dt - desired_dt) > 1e-6:
            raise ValueError(f"HDF5 bin width {dt:.4f}s ≠ {desired_dt}s requested.")

        edges = np.concatenate([time_vec - dt/2, [time_vec[-1] + dt/2]])

        out = {}
        for area in sorted(mapping["area"].unique()):
            rows = mapping[mapping["area"] == area]
            keys = rows["hdf5_key"].values

            if not len(keys):
                continue

            pop_counts = np.zeros_like(time_vec, dtype=float)
            valid = 0
            for k in keys:
                if k in spikes_grp:
                    st = spikes_grp[k][...]
                    if st.size:
                        c, _ = np.histogram(st, bins=edges)
                        pop_counts += c
                        valid += 1

            if not valid:
                if verbose:
                    print(f"{area}: 0 spikes → skipped")
                continue

            out[area] = dict(time=time_vec,
                             total=pop_counts,
                             mean=pop_counts / valid,
                             n_neurons=valid)

            if verbose:
                print(f"{area:>12s}: {valid:3d} neurons, "
                      f"peak total={pop_counts.max():.0f}")

            # optional quick plot 
            if do_plot:
                fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                ax[0].plot(time_vec, pop_counts, color="steelblue")
                ax[0].set_ylabel("Total spikes / bin")
                ax[0].set_title(f"{area}  (n={valid})")

                ax[1].plot(time_vec, pop_counts / valid, color="orchid")
                ax[1].set_ylabel("Mean spikes / neuron / bin")
                ax[1].set_xlabel("Time (s)")

                for ts in footshock_times:
                    for a in ax:
                        a.axvline(ts, color="orange", lw=0.8, alpha=.6)

                fig.suptitle(f"Population activity – Rat{rat_id} – {area}",
                             fontsize=12, y=1.02)
                fig.tight_layout()
                plt.show()

    return out
