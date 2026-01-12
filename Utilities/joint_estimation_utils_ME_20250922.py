# Program: joint_estimation_utils_ME_20250922.py

# Purpose: This script contains functions to jointly estimate a network and its covariates

import networkx as nx
import random
import numpy as np
import scipy.stats as sp
import scipy.special as spec
import scipy.optimize as opt
import scipy.linalg as scilin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import sklearn.metrics as skm
import pandas as pd
import pickle
from group_lasso import LogisticGroupLasso

import gmus_lasso
from gmus_lasso import *


# calcualte the difference between two Gram matrices
def mat_change(orig, new):
    diff = np.linalg.norm((new @ new.T) - (orig @ orig.T), 'fro')**2 / np.linalg.norm((orig @ orig.T), 'fro')**2
    return diff
        


def simulate_data_with_noise(n, k, q, noise_dim, B_mu, B_sigma, a_high, a_range):

    # scaling matrix
    J = np.identity(n) - np.ones((n,n))/n
    J_B = np.identity(k) - np.ones((k,k))/n


    ############################
    ##### Step 1: Generate the degree heterogeneity parameters a = (a1,..,an), where a1, ..., an ~ U[-0.25, 0.75]
    a_low = a_high - a_range
    a = np.random.uniform(low=a_low, high=a_high, size=n)

    ############################
    ##### Step 2: Generate k latent vector centers mu1, ... , muk in R^k with coordinates i.i.d. from U[−1, 1]
    mu = np.random.uniform(low = -1, high = 1, size = (k,k))
    
    ############################
    ##### Step 3: Generate latent variables Z in R^(n x k): 
    ## first generate a matrix Z_0 ∈ R^(n×k) such that each entry is i.i.d. N(0, 1). 
    ## Then we divide n data points equally into k subsets, and for point in each subset, add mu1, ..., mk to them respectively.
    ## Lastly we transform Z by 1) setting Z = JZ, 2) normalizing Z such that |(ZZ^T)|F = n, and 
    ##### 3) rotating Z = ZR for some rotation matrix R such that the covariance of Z is a diagonal matrix;

    Z_0 = np.random.normal(0, 1, size = (n,k))
    subset_size = int(n/k)
    Z_01 = Z_0.copy()

    for i in range(k):
        start = i * subset_size
        stop = start + subset_size
        Z_01[start:stop] += mu[i, :]

    # scale Z by J
    Z_1 = J @ Z_01

    # Compute the Frobenius norm of Z_1 Z_1^T
    frobenius_norm = np.linalg.norm((Z_1 @ Z_1.T), 'fro')

    # Calculate the scaling factor to normalize the Frobenius norm to n
    scaling_factor = np.sqrt(n / frobenius_norm)

    # Scale the matrix Z_1
    Z_1_normalized = Z_1 * scaling_factor

    # Compute the covariance matrix of the normalized Z_1
    cov_matrix = np.cov(Z_1_normalized, rowvar=False)

    # Perform eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Construct the rotation matrix R using the eigenvectors
    R = eigenvectors

    # Rotate the normalized matrix Z_1
    Z_final = Z_1_normalized @ R

    ############################
    #####  Step 4: Generate the coefficients B ∈ R^k×q, with each entry i.i.d from N(0, 1)
    y = np.zeros((1,q))
    if noise_dim > 0:
        not_noise = q-noise_dim
        B_real = np.random.normal(B_mu, B_sigma, size = (k, not_noise))
        B_noise = np.zeros((k, noise_dim))
        B = np.hstack((B_real, B_noise))
    else:
        B = np.random.normal(B_mu, B_sigma, size = (k, q))

    ############################
    #####  Get A and Y
    # Generate A
    a = np.array(a)
    P_a = spec.expit(a[:, np.newaxis] + a + (Z_final @ Z_final.T))
    ## Make sure A is symmetric
    A1 = np.random.binomial(n = 1, p = P_a)
    A = np.triu(A1) + np.triu(A1, k = 1).T
  
    # Generate Y    
    P_y =  (np.ones((n,1)) @ np.array(y,ndmin=2)) + (Z_final @ B)
    Y = np.random.binomial(n = 1, p = spec.expit(P_y))
    
    return(A, Y, B, Z_final, a, y)


##################################################################################
# Estimation code utilities
##################################################################################


def normalize_Z(Z):
    """
    Rotate Z so that (1/n) Z^T Z is diagonal.
    Returns (Z_rot, V, eigvals), where:
      - Z_rot = Z @ V
      - V: eigenvectors of (1/n) Z^T Z (columns), sorted by descending eigenvalues
      - eigvals: sorted eigenvalues (descending)
    """
    n = Z.shape[0]
    # Sample covariance of columns
    C = (Z.T @ Z) / max(n, 1)
    # Symmetric EVD
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]          # sort descending
    eigvals = eigvals[idx]
    V = eigvecs[:, idx]
    Z_rot = Z @ V
    return Z_rot, V, eigvals


def build_theta_A(a, Z):
    """Theta_A = a 1^T + 1 a^T + Z Z^T  (n x n)"""
    return a[:, None] + a[None, :] + Z @ Z.T


def build_theta_Y(y, Z, B):
    """Theta_Y = 1 y^T + Z B  (n x q)"""
    n = Z.shape[0]
    return np.ones((n, 1)) @ np.array(y, ndmin=2) + Z @ B


def logloss(true, theta):
    """
    Bernoulli negative log-likelihood for binary data with *logits* theta.
    Stable form: sum[ log(1 + exp(theta)) - true*theta ] = sum[ logaddexp(0, theta) - true*theta ].
    """
    true = np.asarray(true, dtype=float)
    theta = np.asarray(theta, dtype=float)
    return np.sum(np.logaddexp(0.0, theta) - true * theta)


def auc_A_from_theta(A, theta_A):
    """
    AUC for A. If not enough positives/negatives, return np.nan.
    """
    n = A.shape[0]
    y_true = A.astype(int).ravel()
    y_score = spec.expit(theta_A).ravel()
    # need at least one pos and one neg
    if (y_true.sum() == 0) or (y_true.sum() == y_true.size):
        return np.nan
    try:
        return skm.roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def auc_Y_from_theta(Y, theta_Y):
    """
    AUC for Y flattening all entries. If not enough positives/negatives, return np.nan.
    """
    y_true = Y.astype(int).ravel()
    y_score = spec.expit(theta_Y).ravel()
    if (y_true.sum() == 0) or (y_true.sum() == y_true.size):
        return np.nan
    try:
        return skm.roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def remove_columns_close_to_zero(B, tol=1e-4):
    """
    Remove columns of B whose L2 norm is <= tol. Returns (B_kept, removed_indices).
    """
    norms = np.linalg.norm(B, axis=0)
    keep = norms > tol
    removed = np.where(~keep)[0]
    return B[:, keep], removed


def _stable_tail_ok(seq, k_needed=5, tol=1e-5):
    """
    Return True if last k_needed+1 numeric values exist and all k_needed successive deltas < tol.
    """
    seq = [x for x in seq if isinstance(x, (float, int)) and np.isfinite(x)]
    if len(seq) < (k_needed + 1):
        return False
    tail = seq[-(k_needed + 1):]
    deltas = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
    return all(d < tol for d in deltas)

def auc_Y_macro(Y, theta_Y):
    """
    Macro-AUC across columns: average AUC_j over columns j that have
    both classes in Y[:, j] on this split.
    """
    aucs = []
    for j in range(Y.shape[1]):
        yj = Y[:, j].astype(int)
        if yj.min() == yj.max():    # single-class in this split → skip
            continue
        pj = spec.expit(theta_Y[:, j])
        aucs.append(skm.roc_auc_score(yj, pj))
    return float(np.mean(aucs)) if aucs else np.nan


def select_lambda_auto(Y, Z, lambdas, cv_indicate = False, cv=5,
                               adaptive=None, B_prev=None, gamma_adapt=1.0,
                               random_state=0, se_rule=0, verbose=False):
    """
    Choose λ to minimize log loss with Z frozen.
    Returns (lam_star, summary) where summary has mean/std loss per λ and details per fold.
    """
    
    # if you want to cross validate with random split of nodes (not recommended)
    if cv_indicate == True:
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        loss_list, std_loss, per_lambda = [], [], []

        for lam in lambdas:
            fold_loss = []
            for tr, va in kf.split(Z):
                Z_tr, Z_va = Z[tr], Z[va]
                Y_tr, Y_va = Y[tr], Y[va]

                # Fit once per fold across all columns (optimize_* loops over j)
                gamma_hat, B_hat = optimize_gamma_B_binary_group2(
                    Y=Y_tr, Z=Z_tr, B_prev=B_prev, y_prev=None,
                    tuning_param=lam, adaptive=adaptive, gamma_adapt=gamma_adapt
                )

                theta_Y_val = build_theta_Y(gamma_hat, Z_va, B_hat)
                loss = logloss(Y_va, theta_Y_val)
                fold_loss.append(loss)

            if fold_loss:
                loss_list.append(float(np.mean(fold_loss)))
                std_loss.append(float(np.std(fold_loss, ddof=1)))
            else:
                loss_list.append(np.nan)
                std_loss.append(np.nan)
            per_lambda.append(fold_loss)
            if verbose:
                print(f"λ={lam:g}: loss mean={loss_list[-1]:.4f} ± {std_loss[-1]:.4f} over {len(per_lambda[-1])} folds")
                
        loss_list = np.array(loss_list); std_loss = np.array(std_loss)
        if np.all(np.isnan(loss_list)):
            raise RuntimeError("All λ produced NaN loss. Try different grid or CV folds.")
     
    # naive lasso without cross validation (recommended)
    else:
        loss_list = []
        ic_list = []
        num_remain = []
        for lam in lambdas:
            
            gamma_hat, B_hat = optimize_gamma_B_binary_group2(
                Y=Y, Z=Z, B_prev=B_prev, y_prev=None,
                tuning_param=lam, adaptive=adaptive, gamma_adapt=gamma_adapt)
           
            Bshape = np.shape(B_prev)
           
            theta_Y_val = build_theta_Y(gamma_hat, Z, B_hat)
            loss = logloss(Y, theta_Y_val)
            
            B_keep, removed = remove_columns_close_to_zero(B_hat, tol=1e-6)
            n_active = Bshape[1] - len(removed)
            df = n_active * (Bshape[0] + 1) #number of dimensions + 1 (number of predictors)
            n_eff = Y.shape[0]  # or Y.size
            ic = 2.0 * loss + 2 * df #np.log(n_eff) * df
            
            loss_list.append(loss)
            ic_list.append(ic)
            num_remain.append(n_active)
 

            if verbose:
                print(f"λ={lam:g}: loss mean={loss_list[-1]:.4f},  ic = {ic_list[-1]:.4f}")
                print(f"    removed columns: {removed.tolist()}")
                print(f"    norms of kept: {np.linalg.norm(B_keep, axis=0)}")

        if np.all(np.isnan(loss_list)):
            raise RuntimeError("All λ produced NaN loss. Try different grid.")         


    # 1-SE rule for CV: largest λ within 1 std error of best mean
    if cv_indicate == True and se_rule>0 and np.isfinite(loss_list[best_idx]) and np.isfinite(std_loss[best_idx]):
        best_idx = int(np.nanargmin(loss_list))
        lam_best = lambdas[best_idx]
        thresh = loss_list[best_idx] + se_rule*std_loss[best_idx]
        elig = [l for l, m in zip(lambdas, loss_list) if np.isfinite(m) and m <= thresh]
        lam_star = max(elig) if elig else lam_best
    
    # If no CV, choose largest lambda with close to minimum IC
    else:
        # Convert to NumPy arrays for safe masking / argmin
        ic_arr = np.asarray(ic_list, dtype=float)
        num_remain_arr = np.asarray(num_remain, dtype=int)

        # Not interested in results which remove every column; restrict to those which don't
        mask = num_remain_arr != 0   # boolean array
        print("Lambdas with remaining columns: ", mask)

        if mask.any():
            # indices of lambdas that DO NOT remove all columns
            valid_indices = np.where(mask)[0]
            # pick best IC among valid indices
            best_within_valid = np.nanargmin(ic_arr[valid_indices])
            best_idx = int(valid_indices[best_within_valid])
        else:
            # fallback: no valid lambda, use overall best
            best_idx = int(np.nanargmin(ic_arr))
        
        lam_best = lambdas[best_idx]
        thresh = ic_list[best_idx]#*1.01
        eligible_idx = [i for i in range(len(lambdas)) if np.isfinite(ic_arr[i]) and ic_arr[i] <= thresh and num_remain_arr[i] != 0]

        if eligible_idx:
            # choose the largest λ among the eligible *valid* ones
            lam_star = max(lambdas[i] for i in eligible_idx)
        else:
            # if somehow no eligible valid λ (should be rare), fall back
            lam_star = lam_best

    if cv_indicate == True:
        summary = {
            "loss_list": dict(zip(lambdas, loss_list)),
            "std_loss": dict(zip(lambdas, std_loss)),
            "per_lambda_folds": dict(zip(lambdas, per_lambda)),
            "best_raw": lam_best,
            "selected": lam_star
        }
    else:
        summary = {
            "loss": dict(zip(lambdas, loss_list)),
            "ic": dict(zip(lambdas, ic_list)), 
            "best_raw": lam_best,
            "selected": lam_star}
    return lam_star, summary

###################################################################################
# Optimizer Update for (Z, a)
##################################################################################

def update_step_Za_adam(a, y, Z, B, A, Y, m_Z, v_Z, m_a, v_a, 
                        eta_z, eta_a, t, lamb, beta1=0.9, 
                        beta2=0.999, epsilon=1e-8):
    """
    One Adam step for (Z, a)
      - center Z via J, center a to zero mean.

    Parameters
    ----------
    a : (n,) node intercepts
    y : (q,) Y-intercepts
    Z : (n,k)
    B : (k,q)
    A : (n,n) binary adjacency
    Y : (n,q) binary covariates
    m_Z, v_Z : Adam momentum and velocity for Z (same shapes as Z)
    m_a, v_a : Adam momentum and velocity for a (same shapes as a)
    eta_z, eta_a : step sizes for Z and a
    t : time step (for bias correction)
    beta1, beta2 : Adam hyperparameters
    epsilon : Adam epsilon

    Returns
    -------
    theta_A, theta_Y, Z_new, a_new, m_Z, v_Z, m_a, v_a
    """
    n = Z.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n

    # Linear predictors
    theta_A = build_theta_A(a, Z)      # (n x n)
    theta_Y = build_theta_Y(y, Z, B)   # (n x q)

    # Probabilities
    pA = spec.expit(theta_A)
    pY = spec.expit(theta_Y)

    # A
    SA= (pA - A)

    # Y
    SY = (pY - Y)

    # Gradients (factor 2 to account for symmetric contribution)
    grad_Z = 2.0 * (SA @ Z) + lamb * (SY @ B.T)
    grad_a = 2.0 * (SA @ np.ones(n))

    # Adam updates for Z
    m_Z = beta1 * m_Z + (1 - beta1) * grad_Z
    v_Z = beta2 * v_Z + (1 - beta2) * (grad_Z ** 2)
    
    # Bias correction
    m_Z_hat = m_Z / (1 - beta1 ** t)
    v_Z_hat = v_Z / (1 - beta2 ** t)
    
    Z_new = Z - eta_z * m_Z_hat / (np.sqrt(v_Z_hat) + epsilon)  # Fixed: subtract for descent

    # Adam updates for a
    m_a = beta1 * m_a + (1 - beta1) * grad_a
    v_a = beta2 * v_a + (1 - beta2) * (grad_a ** 2)
    
    # Bias correction
    m_a_hat = m_a / (1 - beta1 ** t)
    v_a_hat = v_a / (1 - beta2 ** t)
    
    a_new = a - eta_a * m_a_hat / (np.sqrt(v_a_hat) + epsilon)  # Fixed: subtract for descent

    # Projection / centering for identifiability & stability
    Z_new = J @ Z_new
    a_new = a_new - np.mean(a_new)

    return theta_A, theta_Y, Z_new, a_new, m_Z, v_Z, m_a, v_a

def update_step_Za_adagrad(a, y, Z, B, A, Y, 
                        eta_z, eta_a, lamb, G_Z, G_a, epsilon=1e-8):
    """
    One adagrad step for (Z, a)
      - center Z via J, center a to zero mean.

    Parameters
    ----------
    a : (n,) node intercepts
    y : (q,) Y-intercepts
    Z : (n,k)
    B : (k,q)
    A : (n,n) binary adjacency
    Y : (n,q) binary covariates
    eta_z, eta_a : step sizes for Z and a
    t : time step (for bias correction)
    epsilon

    Returns
    -------
    theta_A, theta_Y, Z_new, a_new, m_Z, v_Z, m_a, v_a
    """
    n = Z.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n

    # Linear predictors
    theta_A = build_theta_A(a, Z)      # (n x n)
    theta_Y = build_theta_Y(y, Z, B)   # (n x q)

    # Probabilities
    pA = spec.expit(theta_A)
    pY = spec.expit(theta_Y)

    # A
    SA = (A - pA)

    # Y
    SY = (Y - pY)

    # Gradients (factor 2 to account for symmetric contribution)
    grad_Z = 2.0 * (SA @ Z) + lamb * (SY @ B.T)
    grad_a = 2.0 * (SA @ np.ones(n))

    # Update Adagrad accumulators
    G_Z += grad_Z ** 2
    G_a += grad_a ** 2
    
    # Adagrad updates (adaptive per-parameter learning rates)
    Z_temp = Z + (eta_z / np.sqrt(G_Z + epsilon)) * grad_Z
    a_new = a + (eta_a / np.sqrt(G_a + epsilon)) * grad_a

    # Projection / centering for identifiability & stability
    Z_new = J @ Z_temp
    a_new = a_new - np.mean(a_new)

    return Z_new, a_new



##################################################################################
# unpenalized Y-step (no lasso) 
##################################################################################

def optimize_gamma_B_binary(Y, Z):
    """
    Unpenalized per-column logistic regression with intercept.
    Returns (y_hat, B_hat) where y_hat[j] is intercept for column j.
    """
    n, q = Y.shape
    k = Z.shape[1]
    B_new = np.zeros((k, q))
    y_new = np.zeros(q)
    probY = np.zeros((n,q))

    # Use a stable, deterministic solver
    base = LogisticRegression(
        penalty=None,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=5000
    )

    for j in range(q):
        yj = Y[:, j].astype(int)
        # If constant column, keep zeros
        if (yj.sum() == 0) or (yj.sum() == n):
            B_new[:, j] = 0.0
            y_new[j] = np.log((yj.mean() + 1e-8) / (1 - yj.mean() + 1e-8))  # fallback
            continue
        clf = base.fit(Z, yj)
        B_new[:, j] = clf.coef_.ravel()
        y_new[j] = float(clf.intercept_)
    return y_new, B_new


##################################################################################
# group-lasso Y-step (lasso)
##################################################################################

def optimize_gamma_B_binary_group2(Y, Z, B_prev, y_prev, tuning_param, adaptive=None, gamma_adapt=1.0):
    """
    Group logistic lasso per column j with Z features (size n x k).
    One group per column: all k dims grouped together -> column selection.
    Retains intercept properly and extracts binary coefs robustly.

    adaptive: if "Y", use adaptive 1 / (||B_prev[:,j]||^gamma_adapt) as group weight; else None.
    """

    n, q = Y.shape
    k = Z.shape[1]
    B_new = np.zeros((k, q))
    gamma_new = np.zeros(q)

    groups = [1] * k  # single group (column selection)

    for j in range(q):
        yj = Y[:, j].astype(int)

        # Skip degenerate columns
        if (yj.sum() == 0) or (yj.sum() == n):
            B_new[:, j] = 0.0
            gamma_new[j] = np.log((yj.mean() + 1e-8) / (1 - yj.mean() + 1e-8))
            continue

        if adaptive == "Y":
            base_norm = np.linalg.norm(B_prev[:, j])
            group_reg = (1.0 / (base_norm ** gamma_adapt + 1e-8)) * tuning_param
        else:
            group_reg = tuning_param

        gl = LogisticGroupLasso(
            groups=groups,
            group_reg=group_reg,
            l1_reg=0.0,
            supress_warning=True,
            fit_intercept=True,
            n_iter=5000,
            scale_reg="none",
            tol=1e-5
        )
        gl.fit(Z, yj)
        
        # Add in because it keeps giving coefficients for both classes
        intercept_arr = np.asarray(gl.intercept_)
        if len(intercept_arr) == 2:
            # Proper difference: positive class minus negative class
            intercept = float(intercept_arr[1] - intercept_arr[0])
        else:
            intercept = float(intercept_arr.item())
            
        # Handle coefficients
        coef_arr = np.asarray(gl.coef_)
        if coef_arr.ndim == 2 and coef_arr.shape[1] == 2:
            # Shape is (k, 2): columns are [class_0_coefs, class_1_coefs]
            coef = coef_arr[:, 1] - coef_arr[:, 0]  # (k,) vector: class_1 - class_0
        else:
            # Shape is (k,) or (1, k): standard binary classification
            coef = coef_arr.ravel()

        B_new[:, j] = coef
        gamma_new[j] = intercept
    
    
    return gamma_new, B_new


###################################################################################
# Main estimation steps
##################################################################################


def estimation_latent(A, Y, Z_init, B_init, a_init, y_init,
                           eta, lamb, T, 
                           G_Z, G_a, start_T=1, true_avail = False, Z_true = None, B_true = None,
                           tol_loss=1e-10, patience_loss=50, optimizer="adagrad",
                           beta1=0.9, beta2=0.999,
                           verbose=False):
    """
    Pre/post-lasso alternating estimation (Z,a) <-> (y,B) WITHOUT penalty on (y,B) using optimizer.
    Early-stop when loss_A and loss_Y both stabilize (|delta| < tol for 'patience' steps).
    """
    Z = Z_init.copy()
    B = B_init.copy()
    a = a_init.copy()
    y = y_init.copy()

    n, q = Y.shape
    k = Z.shape[1]
    
    eta_min = 0 

    # Adam momentum and velocity accumulators
    m_Z = np.zeros_like(Z)
    v_Z = np.zeros_like(Z)
    m_a = np.zeros_like(a)
    v_a = np.zeros_like(a)

    aucA_hist, aucY_hist, delta_Z, delta_B, delta_LP, lossA_hist, lossY_hist = [], [], [], [], [], [], []
    
    best = {
        "score": np.inf,
        "t": None,
        "Z": None, "a": None, "B": None, "y": None,
        "aucA_hist": None, "aucY_hist": None,
        "lossA_hist": None, "lossY_hist": None,
        "delta_Z": None,
        "delta_B": None,
        "delta_LP": None
    }
    
    if true_avail:
        delta_Z_true, delta_B_true, delta_LP_true = [], [], []
        best['delta_Z_true'] = None
        best['delta_B_true'] = None
        best['delta_LP_true'] = None
        
    print("start_T, T:", start_T, T)
    print("initial best score:", best["score"])

    for t in range(start_T, T + 1):  # Adam uses 1-based indexing for bias correction
        # (Z,a)-step
        Z_prev = Z.copy()
        B_prev = B.copy()
        
        # Cosine annealing for eta
        num = 1.0 + np.cos((t * np.pi) / T)
        eta_t = eta_min + (0.5*(eta - eta_min) * num )
        
        # step sizes as in Zhang 2022: eta_z = eta/||Z||_F^2, eta_a = eta/(2n)
        eta_z = eta_t / (np.linalg.norm(Z, 'fro') ** 2 + 1e-12)
        eta_a = eta_t / (2.0 * n)

        # Use optimizer
        if optimizer.lower() == "adam":
            # We ignore theta_A/theta_Y returned here and recompute later
            _, _, Z, a, m_Z, v_Z, m_a, v_a = update_step_Za_adam(
                a=a, y=y, Z=Z, B=B, A=A, Y=Y,
                m_Z=m_Z, v_Z=v_Z, m_a=m_a, v_a=v_a,
                eta_z=eta_z, eta_a=eta_a, t=t, lamb=lamb,
                beta1=beta1, beta2=beta2, epsilon=1e-8
            )
        elif optimizer.lower() == "adagrad":
            Z, a = update_step_Za_adagrad(
                a=a, y=y, Z=Z, B=B, A=A, Y=Y,
                eta_z=eta_z, eta_a=eta_a, lamb=lamb,
                G_Z=G_Z, G_a=G_a, epsilon=1e-8
            )
        else:
            raise ValueError("optimizer must be 'adam' or 'adagrad'")
        
        # diagonalize (1/n) Z^T Z, rotate B to preserve Θ_Y 
        Z, V, _ = normalize_Z(Z)
        B = V.T @ B

        # (y,B)-step (unpenalized)
        y, B = optimize_gamma_B_binary(Y, Z)

        # diagnostics
        theta_A_now = build_theta_A(a, Z)
        theta_Y_now = build_theta_Y(y, Z, B)
        lossA = logloss(A, theta_A_now)
        lossY = logloss(Y, theta_Y_now)
        aucA= auc_A_from_theta(A, theta_A_now)
        aucY = auc_Y_from_theta(Y, theta_Y_now)
        
        
        aucA_hist.append(aucA)
        aucY_hist.append(aucY)
        lossA_hist.append(lossA)
        lossY_hist.append(lossY)
        
        # look at how the Z and B matrices change over time                             
        delta_Z_step = mat_change(Z_prev, Z)
        delta_Z.append(delta_Z_step)
        
        delta_B_step = mat_change(B_prev, B)
        delta_B.append(delta_B_step)
        
        delta_LP_step = mat_change((Z_prev@B_prev), (Z@B))
        delta_LP.append(delta_LP_step)
        
        # If the true values of Z and B are available, calculate the distance from the truth
        if true_avail:
            delta_Z_step_T = mat_change(Z_true, Z)
            delta_Z_true.append(delta_Z_step_T)

            delta_B_step_T = mat_change(B_true, B)
            delta_B_true.append(delta_B_step_T)

            delta_LP_step_T = mat_change((Z_true@B_true), (Z@B))
            delta_LP_true.append(delta_LP_step_T)
        
        # This isn't a convex optimization, so have a fail safe in case the optimization starts straying
        curr_score = (lossA / (n*(n-1))) + (lamb * lossY / (n*q))
        
        if curr_score < best["score"] + 1e-12:   # tiny epsilon to avoid copy churn
            best["score"] = curr_score
            best["t"] = t
            # snapshot parameters (use .copy() so future updates don’t mutate the saved best)
            best["Z"] = Z.copy()
            best["a"] = a.copy()
            best["B"] = B.copy()
            best["y"] = y.copy()
            # snapshot histories (slice to copy)
            best["aucA_hist"] = aucA_hist[:]
            best["aucY_hist"] = aucY_hist[:]
            best["lossA_hist"] = lossA_hist[:]
            best["lossY_hist"] = lossY_hist[:]
            best["delta_Z"] = delta_Z[:]
            best["delta_B"] = delta_B[:]      
            best["delta_LP"] = delta_LP[:]
            if true_avail:
                best['delta_Z_true'] = delta_Z_true[:]
                best['delta_B_true'] = delta_B_true[:]
                best['delta_LP_true'] = delta_LP_true[:]
                
        # print current status
        if verbose and (t % 1000 == 0):
            print(f"[iter {t:4d}] loss_A={lossA:.4f} loss_Y={lossY:.4f} AUC_A={aucA:.4f} AUC_Y={aucY:.4f} score={curr_score:.4f}")

        # early stopping on loss stability (check for worsening score or stability for a certain number of steps)
        if t > start_T + 5:
            drop_pct = (best["score"] < np.inf) and (curr_score > 1.05 * best["score"])
            if (_stable_tail_ok(lossA_hist, patience_loss, tol_loss) and \
               _stable_tail_ok(lossY_hist, patience_loss, tol_loss) and \
               _stable_tail_ok(delta_LP, patience_loss, 1e-10)) or drop_pct:
                if verbose:
                    if curr_score > 1.05 * best["score"]:
                        print(f"Early stop: score increased by 5%")
        
                    else: 
                        print(f"Early stop: loss stable for {patience_loss} steps (tol={tol_loss}).")
                break
                
    if true_avail:
        return (best["Z"], best["a"], best["B"], best["y"], best["aucA_hist"], best["aucY_hist"], best["lossA_hist"], best["lossY_hist"],
            best["delta_Z"], best["delta_B"], best["delta_LP"], best["t"], best['delta_Z_true'], best['delta_B_true'], best['delta_LP_true'])

    else:
        return (best["Z"], best["a"], best["B"], best["y"], best["aucA_hist"], best["aucY_hist"], best["lossA_hist"], best["lossY_hist"],
            best["delta_Z"], best["delta_B"], best["delta_LP"], best["t"])

##################################################################################
# run simulation 
##################################################################################

def run_simulations(n, k, q, eta, lamb, T,
                         A_in, Y_in,
                         Z_init, B_init, a_init, y_init,
                         tuning_param_list=None, change_lasso_step=None,
                         adaptive=None, gamma_adapt=1.0,
                         beta1=0.9, beta2=0.999,
                         tol_loss=1e-10, patience_loss=50,
                         auto_select_lambda=False,
                         lambda_grid=None, cv_indicate = False, cv_folds=0,
                         se_rule = 0, ME_aware = False, true_avail = False, Z_true = None, B_true = None,
                         random_state=0, optimizer = "adagrad",
                         verbose=False):
    
   

    # Stage 1: initial estimation without penalization
    G_Z = np.zeros_like(Z_init) 
    G_a = np.zeros_like(a_init)
    
    out = estimation_latent(
        A=A_in, Y=Y_in, Z_init=Z_init, B_init=B_init, a_init=a_init, y_init=y_init,
        eta=eta, lamb=lamb,
        T=T if change_lasso_step is None else change_lasso_step,
        G_Z=G_Z, G_a=G_a, start_T=1, true_avail=true_avail, Z_true=Z_true, B_true=B_true,
        optimizer=optimizer, beta1=beta1, beta2=beta2,
        tol_loss=tol_loss, patience_loss=patience_loss,
        verbose=verbose
    )

    if  true_avail:
        (Z1, a1, B1, y1, aucA1, aucY1, lossA1, lossY1,
         delta_Z, delta_B, delta_LP, t, delta_Z_true, delta_B_true, delta_LP_true) = out
    else:
        (Z1, a1, B1, y1, aucA1, aucY1, lossA1, lossY1,
         delta_Z, delta_B, delta_LP, t) = out
        
    pre = { "Z": Z1, "a": a1, "B": B1, "y": y1, "aucA": aucA1, "aucY": aucY1, "lossA": lossA1, "lossY": lossY1, "delta_Z": delta_Z,
            "delta_B": delta_B, "delta_LP": delta_LP}

    if true_avail:
        pre["delta_Z_true"] = delta_Z_true
        pre["delta_B_true"] = delta_B_true
        pre["delta_LP_true"] = delta_LP_true

    results = {"pre": pre, "post": []}

    # If no lasso stage requested, end algorithm
    if change_lasso_step is None:
        return results

    # If lasso is used, determine the optimal value of lambda
    if auto_select_lambda:
        if lambda_grid is None:
            lambda_grid = np.geomspace(1e-3, 1.0, 8)   # default grid
            
        lam_star, cv_summary = select_lambda_auto(
            Y=Y_in, Z=Z1, lambdas=lambda_grid, cv_indicate=cv_indicate, cv=cv_folds,
            adaptive=adaptive, B_prev=B1, gamma_adapt=gamma_adapt,
            random_state=random_state, se_rule=se_rule, verbose=verbose
        )
        lambdas_to_run = [lam_star]
        
    else:
        if tuning_param_list is None or len(tuning_param_list) == 0:
            return results
        cv_summary = None
        lambdas_to_run = tuning_param_list

    # Stage 2: lasso + refinement with chosen lambda
    if verbose:
        print("Stage 2 (lasso + unpenalized refinement):")

    T2 = max(0, T - (t if change_lasso_step is not None else 0))
    Z_fixed, a_fixed = Z1, a1

    for lam in lambdas_to_run:
        if verbose:
            print(f"  - group lasso (λ={lam:g}) with Z frozen")
        gamma_hat, B_hat = optimize_gamma_B_binary_group2(
            Y=Y_in, Z=Z_fixed, B_prev=B1, y_prev=y1,
            tuning_param=lam, adaptive=adaptive, gamma_adapt=gamma_adapt
        )
      
        # Drop zero columns and refine
        B_kept, removed = remove_columns_close_to_zero(B_hat, tol=1e-4)
        Y_kept = np.delete(Y_in, removed, axis=1)
        y_kept = np.delete(gamma_hat, removed)

        if verbose:
            print(f"    removed columns: {removed.tolist()}")

        if B_kept.size == 0:
            results["post"].append({"tuning": lam, "removed": removed, "Z": Z_fixed, "a": a_fixed,
                                    "B": B_kept, "y": y_kept, "aucA": [], "aucY": [], "lossA": [], "lossY": [], "cv_summary": cv_summary, "delta_Z": [],
                                   "delta_B": [],  "delta_LP": []})
            print("No features remaining after lasso - stopping")
            break

        if verbose:
            print("    refinement (unpenalized) with Adam and early stopping...")
        
        # Single-group GMUL refinement (use if measurement-aware setting)
        # Toggle with a simple flag; reuses (gamma_hat, B_hat) as warm start
        if ME_aware:
            delta_grid = np.round(np.arange(0.00, 0.52, 0.02), 2)  # or pass in via args
            if verbose:
                print(f"    GMUL (single-group) refinement: λ={lam:g}, |δ-grid|={len(delta_grid)} (warm start from lasso)")
            y_hat, B_hat = optimize_gamma_B_binary_group2_gmul_single(
                Y=Y_kept, Z=Z_fixed, B_prev=B_kept, y_prev=y_kept,
                tuning_param=lam, delta_grid=delta_grid,
                select_by='logloss',  # or 'auc'
                max_irls=50, max_inner=4000,
                tol_outer=1e-6, tol_inner=1e-6,
                seed=random_state
            )
            if verbose:
                print(f"    GMUL complete")
        else:
            y_hat = y_kept
            B_hat = B_kept
                
        if verbose:
            print(f"    Begin second stage estimating at T = {T2}")

        out = estimation_latent(
            A=A_in, Y=Y_kept,
            Z_init=Z_fixed, B_init=B_hat, a_init=a_fixed, y_init=y_hat,
            eta=eta, lamb=lamb, T=T2,
            G_Z=G_Z, G_a=G_a, start_T=1, true_avail=true_avail, Z_true=Z_true, B_true=B_true,
            optimizer=optimizer, beta1=beta1, beta2=beta2,
            tol_loss=tol_loss, patience_loss=patience_loss,
            verbose=verbose
        )
        
        if true_avail:
            Z2, a2, B2, y2, aucA2, aucY2, lossA2, lossY2, delta_Z, delta_B, delta_LP, t, delta_Z_true, delta_B_true, delta_LP_true = out
        
        else:
            Z2, a2, B2, y2, aucA2, aucY2, lossA2, lossY2, delta_Z, delta_B, delta_LP, t = out
        
        if verbose:
            print(f"    End second stage estimating")
        
        results["post"].append({
            "tuning": lam, "removed": removed, "Z": Z2, "a": a2,
            "B": B2, "y": y2, "aucA": aucA2, "aucY": aucY2, "lossA": lossA2, "lossY": lossY2,
            "cv_summary": cv_summary, "delta_Z": delta_Z, "delta_B": delta_B,
            "delta_LP": delta_LP
        })
        
        if true_avail:
            results["post"].append({"delta_Z_true": delta_Z_true, "delta_B_true": delta_B_true, "delta_LP_true": delta_LP_true})

    return results

