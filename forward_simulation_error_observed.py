def compute_and_plot_fse_observed_AR(model_dir, n_steps=10, plot=True, model_fname="my_model.pkl"):
    """
    FSE in observed space for rSLDS with AR(1) emissions.
    """
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Load model and data
    with open(Path(model_dir) / model_fname, "rb") as f:
        model = pickle.load(f)
    x_hat = np.load(Path(model_dir) / "x_hat.npy")
    FR_z = np.load(Path(model_dir) / "FR_z.npy")

    #AR coefficients for emissions
    try:
        Cs = model.emissions.Cs  # shape: (K, D, N) or (D, N) if K=1
        ds = model.emissions.ds  # shape: (K, N) or (N,) if K=1
    except AttributeError:
        raise ValueError("Could not find AR emission parameters in your model.")

    K = getattr(model, "K", 1)
    D = x_hat.shape[1]
    N = FR_z.shape[1]

    mse_steps = []
    for dt in range(1, n_steps + 1):
        squared_errors = []
        for t0 in range(FR_z.shape[0] - dt):
            x_pred = x_hat[t0].copy()
            z = 0  # If more than 1 state, you can set z_hat[t0] etc
            # roll forward
            for _ in range(dt):
                # AR(1): x_pred = A @ x_pred + b
                if K == 1:
                    A = model.dynamics.As  # shape (D, D)
                    b = model.dynamics.bs  # shape (D,)
                    x_pred = np.dot(A, x_pred) + b
                else:
                    A = model.dynamics.As[z]
                    b = model.dynamics.bs[z]
                    x_pred = np.dot(A, x_pred) + b
            # emission: y_pred = C x_pred + d
            if K == 1:
                y_pred = np.dot(Cs, x_pred) + ds
            else:
                y_pred = np.dot(Cs[z], x_pred) + ds[z]
            error = np.mean((y_pred - FR_z[t0 + dt]) ** 2)
            squared_errors.append(error)
        mse_steps.append(np.mean(squared_errors))
    
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(1, n_steps+1), mse_steps, marker='o')
        plt.xlabel('Δt steps ahead')
        plt.ylabel('Forward Simulation MSE (observed space)')
        plt.title('Forward Simulation Error (Observed Neural Data, AR(1))')
        plt.tight_layout()
        plt.show()
    return mse_steps
