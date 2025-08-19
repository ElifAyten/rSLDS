import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def compute_and_plot_fse(model_dir, n_steps=10, plot=True, model_fname="my_model.pkl"):
    """
    Compute and plot Forward Simulation Error (FSE) for an rSLDS model.
    Args:
        model_dir: Directory with 'rSLDS.pkl', 'x_hat.npy', 'z_hat.npy' saved.
        n_steps:   Number of steps to roll forward.
        plot:      If True, plot the FSE curve.
        model_fname: Filename of the pickled model (default 'rSLDS.pkl')
    Returns:
        mse_steps: Array of mean MSE for 1-step, 2-step, ..., n-step.
    """
    # load model and inferred latents
    with open(Path(model_dir) / model_fname, "rb") as f:
        model = pickle.load(f)
    x_hat = np.load(Path(model_dir) / "x_hat.npy")   # (T, latent_dim)
    z_hat = np.load(Path(model_dir) / "z_hat.npy")   # (T,)

    # for each t, roll forward latent using AR(1) parameters for the most likely state
    mse_steps = []
    T = x_hat.shape[0]
    for dt in range(1, n_steps + 1):
        errors = []
        for t0 in range(T - dt):
            x_pred = x_hat[t0].copy()
            z_cur = z_hat[t0]
            for d in range(dt):
                # use AR(1) dynamics for the most likely z
                A = model.dynamics.As[z_cur]
                b = model.dynamics.bs[z_cur]
                x_pred = A @ x_pred + b
                # update discrete state if you want (z_cur = z_hat[t0 + d + 1]) -- optional
            errors.append(np.mean((x_pred - x_hat[t0 + dt]) ** 2))
        mse_steps.append(np.mean(errors))

    if plot:
        plt.plot(np.arange(1, n_steps + 1), mse_steps, marker='o')
        plt.xlabel('Δt steps ahead')
        plt.ylabel('Forward Simulation MSE (latents)')
        plt.title('Forward Simulation Error (FSE)')
        plt.show()
    return mse_steps

