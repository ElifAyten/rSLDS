# rSLDS model reprository
This repository contains several Python modules (`name.py` files) with reusable functions for different tasks 
(utilities and scripts for dataset prep, dimensionality reduction, modeling, cross-validation, and evaluation 
(such as mutual information and forward simulation error) on neural/behavioral data.).


# Setup / Installations
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
Note: python 3.1 works the best with this model.



# Example workflow
python load_dataset.py --data path/to/data.h5
python match_data_with_metadata.py --data data.h5 --meta metadata.csv --key subject_id
python normalize_firing_rates.py --in merged.h5 --method zscore
python pca.py --in normalized.h5 --n-components 10
python modelling.py --config configs/model.yaml
python forward_simulation_error.py --model runs/model.pt --data normalized.h5
python mutual_information.py --latents latents.npy --signals pupil.csv speed.csv


# Dataset preparation

create_sub_data.py – Export wide/long firing-rate tables per brain macro-area, with filtering by foot-shock response class and minimum firing rate.
create_sub_data_2.py – Build responsive-neuron tables (raw & z-scored rates + metadata) for a session, with normalized speed and binary foot-shock vector.
create_sub_data_3.py – Export wide/long firing-rate tables per macro-area, keeping only neurons above a mean firing-rate threshold.
load_dataset.py – load_rat_data: Load a rat session from HDF5 (time, firing rates, footshock times, and optional speed/pupil).
match_data_with_metadata.py – match_units_to_hdf5: Map units.csv rows to HDF5 firing-rate row indices by matching cluster keys.
normalize_firing_rates.py – normalize_firing_rates: Z-score each neuron’s firing-rate vector; optionally return the fitted StandardScaler.

# Dimensionality reduction

pca.py –
• pca_summary: Run PCA on firing rates and return cumulative variance, and elbow.
• plot_pca_cumsum: Plot cumulative explained variance with elbow/thresholds.

# Modeling & cross-validation

modelling.py – fit_single_rslds: Fit an input-driven rSLDS (AR emissions) to one session; saves model artifacts (.pkl, x_hat.npy, z_hat.npy, elbo.png and so on).
cross_validation.py – Run K-fold cross-validation for rSLDS models with auto-PCA dimensionality selection.
k_fold_splitter.py – contiguous_kfold_indices: Generate contiguous (time-block) K-fold splits for time-series cross-validation.

# Evaluation & analysis

forward_simulation_error.py – Compute and plot forward simulation error (FSE) of rSLDS latent dynamics across multi-step predictions.
forward_simulation_error_observed.py – compute_and_plot_fse_observed_AR: Compute multi-step FSE in observed space using AR(1) emissions; optionally plot MSE vs. horizon.
mutual_information.py – Functions to compute mutual information (MI) between latent states and behavioral signals, with surrogate tests and pre-/post-shock comparison.
mi_pre_post.py – mi_pre_post_plot: Run MI analysis pre- vs. post-footshock with shuffles/FDR; return results table and bar plot.
switch_statistics.py –
• switch_statistics: Compute pre-/post-footshock switch counts, rates, and occupancy.
• plot_switch_summary: Bar-chart summary of switch statistics.

# Plotting

plot_discrete_states.py – plot_discrete_states: Plot cruns of discrete states over time with optional foot-shock markers.
plot_discrete_states2.py – plot_discrete_states2: Variant for plotting sicrete states with optional markers changed size.
plot_latents_footshock.py – _plot_latents, plot_latents_from_dir, load_and_plot_latents: Plot latent trajectories with foot-shock markers; supports integration and smoothing.
plot_latents_pupil.py – plot_latents_pupil_shocks: Overlay pupil trace with latent trajectories and foot-shock markers.
plot_latents_speed.py – plot_latents_speed_shocks: Same as above, but overlays speed trace.

# Population-level activity
population_activity_region.py – population_activity_by_region: Bin spikes per neuron and aggregate activity (total/mean) per brain area, with optional plots.


