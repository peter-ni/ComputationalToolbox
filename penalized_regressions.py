import numpy as np


# ======= LASSO, SCAD, MCP =======

def LASSOtrainer(sim, lambd, max_iter=10000, tol=1e-3, warmstart=None):
    X, Y = sim['X_scaled'], sim['Y_scaled']
    n, d = sim['inits']['n'], sim['inits']['d']
    beta = warmstart.copy()
    
    def soft_thresholding(z, lambd):
        return np.sign(z) * np.maximum(np.abs(z) - lambd, 0)
    
    for iter in range(max_iter):
        beta_old = beta.copy()
        r = Y - X @ beta
        
        for j in range(d):
            zj = beta[j] + (1 / n) * np.sum(X[:, j] * r)
            beta_j_new = soft_thresholding(zj, lambd)
            r += X[:, j] * (beta[j] - beta_j_new)
            beta[j] = beta_j_new
        
        if np.linalg.norm(beta - beta_old) / np.linalg.norm(beta) < tol or np.linalg.norm(beta) == 0:
            break
    
    if iter == max_iter - 1:
        print("Warning: Maximum number of iterations reached.")
    
    return beta


def SCADtrainer(sim, lambd, gamma=3.7, max_iter=10000, tol=1e-3, warmstart=None):
    X, Y = sim['X_scaled'], sim['Y_scaled']
    n, d = sim['inits']['n'], sim['inits']['d']
    beta = warmstart.copy()
    
    def soft_thresholding(z, lambd):
        if z < -lambd:
            return z + lambd
        elif z > lambd:
            return z - lambd
        else:
            return 0
    
    def f_SCAD(z, lambd, gamma):
        if abs(z) <= 2 * lambd:
            return soft_thresholding(z, lambd)
        elif 2 * lambd < abs(z) <= gamma * lambd:
            updated_z = (gamma - 1) / (gamma - 2) * soft_thresholding(z, gamma * lambd / (gamma - 1))
            return 0 if np.sign(z) != np.sign(updated_z) else updated_z
        else:
            return z
    
    for iter in range(max_iter):
        beta_old = beta.copy()
        r = Y - X @ beta
        
        for j in range(d):
            zj = beta[j] + (1 / n) * np.sum(X[:, j] * r)
            beta_j_new = f_SCAD(zj, lambd, gamma)
            r += X[:, j] * (beta[j] - beta_j_new)
            beta[j] = beta_j_new
        
        if np.linalg.norm(beta - beta_old) / np.linalg.norm(beta) < tol or np.linalg.norm(beta) == 0:
            break
    
    if iter == max_iter - 1:
        print("Warning: Maximum number of iterations reached.")
    
    return beta


def MCPtrainer(sim, lambd, gamma, max_iter=10000, tol=1e-3, warmstart=None):
    X, Y = sim['X_scaled'], sim['Y_scaled']
    n, d = sim['inits']['n'], sim['inits']['d']
    beta = warmstart.copy()
    
    XtY = X.T @ Y
    XtX = X.T @ X
    
    for iter in range(max_iter):
        beta_old = beta.copy()
        
        for j in range(d):
            rho = XtY[j] - np.sum(XtX[j, np.arange(d) != j] * beta[np.arange(d) != j])
            x_norm = np.sum(X[:, j]**2)
            
            if rho < -lambd / 2:
                beta[j] = (rho + lambd / (1 - 1 / gamma)) / x_norm
            elif rho > lambd / 2:
                beta[j] = (rho - lambd / (1 - 1 / gamma)) / x_norm
            else:
                beta[j] = 0
            
            beta[j] = np.sign(beta[j]) * max(np.abs(beta[j]) - lambd / (1 - 1 / gamma), 0)
        
        if np.linalg.norm(beta - beta_old) / np.linalg.norm(beta) < tol or np.linalg.norm(beta) == 0:
            break
    
    if iter == max_iter - 1:
        print("Warning: Maximum number of iterations reached.")
    
    return beta


# ======= Multinomial Logistic Regression Gradient Descent with Option for Nesterov Acceleration =======


def l2mlr_objEval(betas, X, Y, lambd):
    K = int(np.max(Y))
    n = X.shape[0]
    Y = Y.reshape(-1, 1)
    
    term1 = 0
    term2 = 0
    term3 = 0
    
    for l in range(K):
        term1 += (lambd / 2) * np.linalg.norm(betas[:, l], ord=2)**2
    
    for i in range(n):
        inner_sum1 = 0
        inner_sum2 = 0
        for l in range(K):
            if Y[i, 0] == l + 1:
                inner_sum1 += X[i, :] @ betas[:, l]
            inner_sum2 += np.exp(X[i, :] @ betas[:, l])
        
        term2 += (1 / n) * inner_sum1
        term3 += (1 / n) * np.log(inner_sum2)
    
    return term1 - term2 + term3


def l2mlr_descent(X, Y, lambd, accel, tol=1e-3, max_iters=1000, alpha=1):
    if lambd <= 0:
        print("Choose lambda > 0")
        return
    if tol <= 0:
        print("Choose tol > 0")
        return
    if max_iters <= 0 or not isinstance(max_iters, int):
        print("Must choose an integer max_iters > 0")
        return

    n, d = X.shape
    K = int(np.max(Y))
    X = np.hstack([np.ones((n, 1)), X])
    
    betas = np.zeros((d + 1, K))
    f_hat = []

    t = 0
    converged = False

    while t <= max_iters and not converged:
        grads = np.zeros((d + 1, K))
        
        if t > 0:
            prev_prev_betas = prev_betas.copy()
        prev_betas = betas.copy()
        
        for l in range(K):
            ind_class_l = np.mean(X[Y == l+1], axis=0).reshape(-1, 1)
            
            prob_sum = np.zeros(d + 1)
            probs = np.zeros(n)
            
            for i in range(n):
                exp_sum = np.sum(np.exp(X[i, :] @ betas))
                probs[i] = np.exp(X[i, :] @ betas[:, l]) / exp_sum
                prob_sum += (probs[i] * X[i, :]) / n
            
            if not accel:
                grads[:, l] = lambd * betas[:, l] - ind_class_l.flatten() + prob_sum
                betas[:, l] -= alpha * grads[:, l]
            else:
                if t == 0:
                    lookahead = betas[:, l]
                else:
                    lookahead = prev_betas[:, l] + (t - 1) / (t + 2) * (prev_betas[:, l] - prev_prev_betas[:, l])
                
                grads[:, l] = lambd * lookahead - ind_class_l.flatten() + prob_sum
                betas[:, l] = lookahead - alpha * grads[:, l]
        
        f_hat.append(l2mlr_objEval(betas=betas, X=X, Y=Y, lambd=lambd))
        t += 1
        
        if t >= 2 and abs(f_hat[-1] - f_hat[-2]) < tol:
            converged = True

    return {'betas': betas, 'f_hat': f_hat}




# ======= ISTA and FISTA (Proximal Gradient Descent) Methods for LASSO =======

def ISTAeval(X_s, Y_s, beta_s, lambd):
    n = X_s.shape[0]
    residual_norm = np.linalg.norm(Y_s - X_s @ beta_s, 2) ** 2
    l1_norm = lambd * np.linalg.norm(beta_s, 1)
    return (1 / (2 * n)) * residual_norm + l1_norm


def elementwise_soft_thresh(u, alpha, lambd):
    out = np.copy(u)
    for i in range(len(u)):
        if u[i] > lambd * alpha:
            out[i] = u[i] - lambd * alpha
        elif -lambd * alpha <= u[i] <= lambd * alpha:
            out[i] = 0
        else:
            out[i] = u[i] + lambd * alpha
    return out



def prox_LASSO_descent(X, Y, lambd, alpha=0.05, accel=False, tol=1e-4, max_iter=1000):
    n, d = X.shape
    
    X_col_means = np.mean(X, axis=0)
    X_col_sd = np.std(X, axis=0) * np.sqrt((n - 1) / n)
    X_scaled = (X - X_col_means) / X_col_sd

    Y_mean = np.mean(Y)
    Y_scaled = Y - Y_mean

    beta_scaled = np.zeros((d, 1))
    fs_hat = []
    k = 0
    converged = False
    beta_scaled_prev = np.zeros((d, 1))
    beta_scaled_prev_prev = np.zeros((d, 1))

    while k <= max_iter and not converged:
        if not accel:
            u = beta_scaled - alpha * (1 / n) * X_scaled.T @ (X_scaled @ beta_scaled - Y_scaled)
            beta_scaled = elementwise_soft_thresh(u, alpha, lambd)

            fs_hat.append(ISTAeval(X_scaled, Y_scaled, beta_scaled, lambd))
            if k >= 2 and abs(fs_hat[k] - fs_hat[k-1]) < tol:
                converged = True
            k += 1

        else:
            if k <= 1:
                fs_hat.append(ISTAeval(X_scaled, Y_scaled, beta_scaled_prev_prev, lambd))
            else:
                v = beta_scaled_prev + (k - 2) / (k + 1) * (beta_scaled_prev - beta_scaled_prev_prev)
                u = v - alpha * (1 / n) * X_scaled.T @ (X_scaled @ v - Y_scaled)
                beta_scaled = elementwise_soft_thresh(u, alpha, lambd)
                fs_hat.append(ISTAeval(X_scaled, Y_scaled, beta_scaled, lambd))

                beta_scaled_prev_prev = beta_scaled_prev.copy()
                beta_scaled_prev = beta_scaled.copy()

            if k > 2 and abs(fs_hat[k] - fs_hat[k-1]) < tol:
                converged = True
            k += 1

    # Compute beta0 and de-standardized beta
    beta0 = Y_mean - np.sum(X_col_means * (beta_scaled.flatten() / X_col_sd))
    beta = beta_scaled.flatten() / X_col_sd

    return {
        'beta0': beta0,
        'beta': beta,
        'fs_hat': fs_hat,
        'X_scaled': X_scaled,
        'Y_scaled': Y_scaled,
        'X_col_sd': X_col_sd
    }



