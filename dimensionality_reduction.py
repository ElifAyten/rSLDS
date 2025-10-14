import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, NMF

# expects these to exist in your codebase:
# _std_nan_robust, make_embedding, choose_latent_dim

def pick_embedding_and_latent_dim(
    FR_sec,
    *,
    methods=("pca","fa","ica","nmf"),
    n_components=None,            # None → method's default (min(10, N))
    random_state=0,
    strategy="input_dim_cap",     # or "rule_of_thumb" or "fixed"
    fixed=None,
    cap=20,
    rule_mult=0.4
):
    """
    Returns (best_method, Z_best, latent_dim, meta_dict).
    Chooses among PCA/FA/ICA/NMF by minimizing reconstruction error
    (or maximizing FA log-likelihood), then computes latent_dim using
    choose_latent_dim() WITHOUT variance goals.
    """
    X = _std_nan_robust(FR_sec)

    scores = []
    candidates = []

    for m in methods:
        Z, meta = make_embedding(X, method=m, n_components=n_components,
                                 random_state=random_state, allow_2d_input=False)
        d = Z.shape[1]

        if m == "fa":
            model = meta.get("model", None)
            if model is None:
                model = FactorAnalysis(n_components=d, random_state=random_state).fit(X)
            score = model.score(X)   # mean log-like per sample; higher is better
            crit = -score            # convert to "lower is better"
        else:
            if m == "pca":
                model = PCA(n_components=d, random_state=random_state).fit(X)
                X_rec = model.inverse_transform(model.transform(X))
            elif m == "ica":
                model = FastICA(n_components=d, random_state=random_state, max_iter=2000).fit(X)
                try:
                    X_rec = model.inverse_transform(model.transform(X))
                except Exception:
                    X_rec = (model.transform(X) @ model.mixing_.T) + model.mean_
            elif m == "nmf":
                Xpos = X - X.min() + 1e-6
                model = NMF(n_components=d, init="nndsvda", random_state=random_state, max_iter=2000).fit(Xpos)
                X_rec = model.inverse_transform(model.transform(Xpos))
                X_rec = X_rec + (X.min() - 1e-6)
            else:
                continue
            crit = float(np.mean((X - X_rec)**2))  # lower is better

        scores.append((crit, m))
        candidates.append((m, Z, meta))

    best_idx = int(np.argmin([c for c, _ in scores]))
    best_method = scores[best_idx][1]
    _, Z_best, meta_best = candidates[best_idx]

    latent_dim = choose_latent_dim(
        Z_best, FR_sec,
        strategy=strategy, fixed=fixed, cap=cap, rule_mult=rule_mult
    )
    latent_dim = int(max(1, min(latent_dim, Z_best.shape[1])))

    meta_out = {
        "chosen_method": best_method,
        "score_table": [{ "method": m, "criterion": float(c) } for c, m in scores],
        "embedding_dim": int(Z_best.shape[1]),
        "latent_dim": int(latent_dim),
        "strategy": strategy,
        "cap": cap,
        "rule_mult": rule_mult,
    }
    return best_method, Z_best.astype(np.float32), latent_dim, meta_out
