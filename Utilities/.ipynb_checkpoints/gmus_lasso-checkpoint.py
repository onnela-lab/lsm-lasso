# Program: gmus_lasso.py
# Purpose: GMUL (single-group) for logistic regression with warm start 

# This is a Python adaption of the R Code available at https://github.com/osorensen/hdme to support the publication:
#    Sørensen Hellton KH, Frigessi A, Thoresen M. Covariate Selection in High-Dimensional Generalized Linear Models With Measurement Error.
#    Journal of Computational and Graphical Statistics. 2018;27(4):739–749.

import numpy as np
import scipy.special as spec


def _sigmoid_sg(x):
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out

def _dlogit_sg(x):
    p = _sigmoid_sg(x)
    return p * (1.0 - p)

def _standardize_cols_sg(X):
    means = X.mean(axis=0)
    Xc = X - means
    scales = Xc.std(axis=0, ddof=0)
    scales[(scales == 0) | ~np.isfinite(scales)] = 1.0
    return Xc / scales, means, scales

def _power_L_sg(X, iters=30):
    # Lipschitz for (1/(2n))||Xb - y||^2  is lambda_max( (X^T X)/n )
    n, p = X.shape
    v = np.random.randn(p)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(iters):
        v = X.T @ (X @ v)
        v /= np.linalg.norm(v) + 1e-12
    return float(max((v @ (X.T @ (X @ v))) / max(n, 1), 1e-12))

def _group_soft_threshold_vector(v, tau):
    nrm = np.linalg.norm(v)
    if nrm <= tau:
        return np.zeros_like(v)
    return (1.0 - tau / nrm) * v

def gmul_single_group_logistic(
    W, y, lam, delta_grid,
    init_beta=None, init_intercept=None,
    max_irls=100, max_inner=4000,
    tol_outer=1e-6, tol_inner=1e-6,
    use_backtracking=True, seed=0):
    """
    Single-group GMU lasso (logistic):
        min  NLL_logit(W, y; b0, beta)
             + lam * ||beta||_2
             + delta * ||beta||_2^2

    - W: (n,k) design (use Z here; no intercept column)
    - Warm start with (init_beta, init_intercept) on original scale is supported.
    - Returns a dict with 'intercepts' and 'betas' for each delta in delta_grid (original scale).
    """
    rng = np.random.default_rng(seed)
    W = np.asarray(W, float)
    y = np.asarray(y, float)
    n, p = W.shape

    # Standardize W internally
    Xs, means, scales = _standardize_cols_sg(W)

    # Map warm start (original scale) -> standardized space
    if init_beta is not None:
        beta0_std = init_beta * scales              # beta_orig = beta_std / scales
        if init_intercept is not None:
            b00_std = init_intercept + np.sum((means / scales) * beta0_std)
        else:
            b00_std = 0.0
        b0 = float(b00_std)
        beta = beta0_std.copy()
    else:
        b0 = 0.0
        beta = np.zeros(p)

    deltas = np.atleast_1d(np.array(delta_grid, float))
    B0_std = np.zeros_like(deltas)
    B_std = np.zeros((deltas.size, p))
    converged = np.zeros(deltas.size, dtype=bool)

    for di, d in enumerate(deltas):
        b0_d = b0
        beta_d = beta.copy()

        # IRLS outer loop
        ok_outer = False
        for _ in range(max_irls):
            eta = b0_d + Xs @ beta_d
            mu = _sigmoid_sg(eta)
            V = _dlogit_sg(eta)
            V = np.clip(V, 1e-8, None)

            # Working response + weighted design
            z = eta + (y - mu) / V
            sqrtV = np.sqrt(V)
            Xwf = (sqrtV[:, None]) * Xs       # (n,p)
            yw  = sqrtV * z                   # (n,)
            Xw0 = sqrtV                       # intercept "column" (vector)

            # Lipschitz constant for smooth part:
            # f(beta) = (1/(2n))||Xwf beta - r||^2 + d * ||beta||_2^2
            # L = lambda_max((Xwf^T Xwf)/n) + 2d
            L_data = _power_L_sg(Xwf)
            L = L_data + 2.0 * d
            t = 1.0 / L
            bt = 2.0

            # Inner prox-gradient on beta with b0 held fixed
            r = yw - Xw0 * b0_d
            for _k in range(max_inner):
                resid = (Xwf @ beta_d) - r
                grad_smooth = (Xwf.T @ resid) / n + 2.0 * d * beta_d
                u = beta_d - t * grad_smooth

                beta_new = _group_soft_threshold_vector(u, lam * t)

                if use_backtracking:
                    # f(beta) = (1/(2n))||Xwf b - r||^2 + d||b||^2
                    def f(b):
                        rr = Xwf @ b - r
                        return 0.5 * (rr @ rr) / n + d * (b @ b)

                    lhs = f(beta_new)
                    diff = beta_new - beta_d
                    rhs = f(beta_d) + grad_smooth @ diff + 0.5 * (1.0 / t) * (diff @ diff)
                    tries = 0
                    while lhs > rhs and tries < 20:
                        t /= bt
                        u = beta_d - t * grad_smooth
                        beta_new = _group_soft_threshold_vector(u, lam * t)
                        lhs = f(beta_new)
                        diff = beta_new - beta_d
                        rhs = f(beta_d) + grad_smooth @ diff + 0.5 * (1.0 / t) * (diff @ diff)
                        tries += 1

                if np.linalg.norm(beta_new - beta_d) < tol_inner * max(1.0, np.linalg.norm(beta_d)):
                    beta_d = beta_new
                    break
                beta_d = beta_new

            # Refresh intercept by weighted least squares closed form
            numer = Xw0 @ (yw - Xwf @ beta_d)
            denom = Xw0 @ Xw0
            b0_new = numer / max(denom, 1e-12)

            # IRLS convergence
            if (np.linalg.norm(beta_d - beta) + abs(b0_new - b0_d)) < tol_outer * max(1.0, np.linalg.norm(beta) + abs(b0_d)):
                b0_d = b0_new
                ok_outer = True
                break

            b0_d = b0_new
            beta = beta_d.copy()

        B0_std[di] = b0_d
        B_std[di, :] = beta_d
        converged[di] = bool(ok_outer)

        # warm start next delta
        b0 = b0_d
        beta = beta_d.copy()

    # Map back to original scale
    Betas = B_std / scales[None, :]
    Intercepts = B0_std - (B_std * (means / scales)[None, :]).sum(axis=1)

    return {
        "intercepts": Intercepts,   # (len(delta),)
        "betas": Betas,             # (len(delta), p)
        "deltas": deltas,
        "lambda": float(lam),
        "means": means,
        "scales": scales,
        "converged": converged,
    }


# Single-group GMUL wrapper: per-column fit with warm start

def optimize_gamma_B_binary_group2_gmul_single(
    Y, Z, B_prev, y_prev,
    tuning_param,                 # lambda
    delta_grid,                   # iterable of delta values to try
    select_by="logloss",          # "logloss" or "auc" (train metric)
    max_irls=50, max_inner=4000,
    tol_outer=1e-6, tol_inner=1e-6,
    seed=0):
    """
    Single-group GMUL per column: all k dims penalized as one group.
    Warm starts from (B_prev[:,j], y_prev[j]) when provided.
    Returns (gamma_hat, B_hat) after picking the best delta (per column).
    """
    n, q = Y.shape
    k = Z.shape[1]
    gamma_hat = np.zeros(q)
    B_hat = np.zeros((k, q))

    for j in range(q):
        yj = Y[:, j].astype(int)

        # Degenerate column -> intercept only
        if (yj.sum() == 0) or (yj.sum() == n):
            gamma_hat[j] = np.log((yj.mean() + 1e-8) / (1 - yj.mean() + 1e-8))
            B_hat[:, j] = 0.0
            continue

        init_beta = None if B_prev is None else B_prev[:, j]
        init_intc = None if y_prev is None else float(y_prev[j])

        fit = gmul_single_group_logistic(
            W=Z, y=yj, lam=tuning_param, delta_grid=delta_grid,
            init_beta=init_beta, init_intercept=init_intc,
            max_irls=max_irls, max_inner=max_inner,
            tol_outer=tol_outer, tol_inner=tol_inner,
            use_backtracking=True, seed=seed
        )

        # choose best delta by metric
        inter = fit["intercepts"]; betas = fit["betas"]
        if select_by == "auc":
            scores = []
            for di in range(len(inter)):
                p = spec.expit(inter[di] + Z @ betas[di])
                try:
                    scores.append(skm.roc_auc_score(yj, p))
                except Exception:
                    scores.append(0.5)
            best = int(np.nanargmax(np.array(scores)))
        else:
            losses = []
            for di in range(len(inter)):
                p = np.clip(spec.expit(inter[di] + Z @ betas[di]), 1e-12, 1 - 1e-12)
                losses.append(-np.mean(yj * np.log(p) + (1 - yj) * np.log(1 - p)))
            best = int(np.nanargmin(np.array(losses)))

        gamma_hat[j] = inter[best]
        B_hat[:, j] = betas[best]

    return gamma_hat, B_hat
