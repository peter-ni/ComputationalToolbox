import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd


def sparsePCA(X, nloadings, lambda1, lambda2, scale = True, tol = 1e-4, max_iter=1000):
    n, d = X.shape
    X = X - np.mean(X, axis=0)
    if scale:
        X = StandardScaler().fit_transform(X)
    XtX = X.T @ X
    lambda1_sorted = np.sort(lambda1)[::-1]
    ordered_lambda1 = lambda1_sorted
    V = []
    reconstruction_error = []
    for l in lambda1_sorted:
        A = np.random.randn(d, nloadings)
        B = np.random.randn(d, nloadings)
        for t in range(max_iter):
            B_prev = B.copy()
            if np.isinf(lambda2):
                for j in range(nloadings):
                    pa = XtX @ A[:, j]
                    vec = np.abs(pa) - l / 2
                    vec[vec < 0] = 0
                    B[:, j] = vec * np.sign(pa)
            else:
                for j in range(nloadings):
                    xi = l / (l + lambda2) 
                    elastic_net = ElasticNet(alpha = l + lambda2, l1_ratio = xi, fit_intercept = False)
                    elastic_net.fit(X, X @ A[:, j])
                    B[:, j] = elastic_net.coef_
            U, Sigma, Vt = np.linalg.svd(XtX @ B, full_matrices = False)
            A = U @ Vt
            if np.linalg.norm(B - B_prev) < tol:
                break
        V_l = B / np.linalg.norm(B)
        error = np.linalg.norm(X - X @ V_l @ V_l.T) ** 2
        V.append(V_l)
        reconstruction_error.append(error)
    return ordered_lambda1, V, reconstruction_error


# ======= Matrix Completion with Soft/Hard Imputation =======

def complete_matrix(M, impute, lambda_vals, rank_vals, tol=1e-4, max_iter=10000):
    # Initialize projection of M onto the set Omega
    P_omega_M = np.copy(M)
    P_omega_M[np.isnan(P_omega_M)] = 0
    
    Z_list = []
    m, n = M.shape

    if impute == "SoftImpute":
        ordered_hyperparams = sorted(lambda_vals, reverse=True)
    elif impute == "HardImpute":
        ordered_hyperparams = sorted(rank_vals, reverse=True)
    else:
        raise ValueError("Impute method must be 'SoftImpute' or 'HardImpute'")

    for l in range(len(ordered_hyperparams)):
        if l == 0:
            Z_new = np.zeros((m, n))
        else:
            Z_new = Z_list[l - 1]
        
        converged = False
        k = 1

        while k <= max_iter and not converged:
            # Store Z_new to Z_old
            Z_old = np.copy(Z_new)

            # Compute projection of Z_old onto set Omega
            P_omega_Z = np.copy(Z_old)
            P_omega_Z[np.isnan(M)] = 0
            P_omega_Z_perp = Z_old - P_omega_Z

            # Perform SVD
            SVD_mat = svd(P_omega_M + P_omega_Z_perp, full_matrices=False)
            U, D, Vt = SVD_mat

            if impute == "SoftImpute":
                D = np.maximum(D - ordered_hyperparams[l], 0)
                Z_new = U @ np.diag(D) @ Vt
            elif impute == "HardImpute":
                rank = ordered_hyperparams[l]
                D = D[:rank]
                U = U[:, :rank]
                Vt = Vt[:rank, :]
                Z_new = U @ np.diag(D) @ Vt

            # Check convergence
            diff = np.linalg.norm(Z_new - Z_old, 'fro')**2 / (np.linalg.norm(Z_old, 'fro')**2 + 1e-6)
            if diff <= tol:
                converged = True

            k += 1
        
        Z_list.append(Z_new)

    return Z_list


