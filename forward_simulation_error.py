def compute_fse(model, x_hat, u, deltas=(1,5,10)):
    """
    model: trained ssm.SLDS model
    x_hat: inferred latent states (T, D)
    u: input (T, M)
    deltas: list of step sizes (how far to roll out)
    Returns: dict of {delta: [MSE...]}
    """
    fse = {d: [] for d in deltas}
    for d in deltas:
        for t in range(len(x_hat) - d):
            # Roll forward for d steps using model dynamics
            x_pred = x_hat[t].copy()
            for step in range(d):
                # This should use the model's transition/dynamics (adapt if needed!)
                x_pred = model.dynamics.sample(x_pred[None], u[t+step][None])[0]
            mse = np.mean((x_pred - x_hat[t+d])**2)
            fse[d].append(mse)
    return fse
