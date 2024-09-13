import numpy as np


# ======= Mini-batch SGD for SVM =======

def svm_gradEval(x, y, C, beta):
    if 1 - y * (x @ beta) > 0:
        grad = beta - C * y * x
    else:
        grad = beta
    return grad


def svm_objEval(X, Y, C, beta):
    n = X.shape[0]
    f_sum = 0
    for i in range(n):
        f_sum += max(0, 1 - Y[i] * (X[i, :] @ beta))
    
    f = 0.5 * np.linalg.norm(beta, 2)**2 + C * (1/n * f_sum)
    return f


def batchSGD(X, Y, C, batch_size=64, num_epochs=50, epsilon0=0.2, decay=0.1, accel=False, momentum=0.9):
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ValueError("Must choose integer num_epochs > 0.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Must choose integer batch_size > 0.")
    if decay <= 0:
        raise ValueError("Must choose decay > 0.")
    
    step_size = epsilon0
    n, d = X.shape
    M = int(np.ceil(n / batch_size))
    X = np.hstack([np.ones((n, 1)), X])  # Adding intercept term
    beta = np.zeros(d + 1)
    v = np.zeros(d + 1)
    f_hat = np.zeros(num_epochs)

    for t in range(num_epochs):
        ids = np.random.permutation(n)
        batch_ids = np.array_split(ids, M)
        
        for k in range(M):
            batch_X = X[batch_ids[k], :]
            batch_Y = Y[batch_ids[k]].reshape(-1, 1)
            
            batch_grad = np.zeros(beta.shape)
            for i in range(batch_X.shape[0]):
                batch_grad += svm_gradEval(batch_X[i, :], batch_Y[i], C, beta)
            
            if not accel:
                beta -= step_size / batch_X.shape[0] * batch_grad
            else:
                v = momentum * v - step_size / batch_X.shape[0] * batch_grad
                beta += v
        
        step_size = epsilon0 / (1 + decay * t)
        f_hat[t] = svm_objEval(X, Y, C, beta)
    
    return {'beta': beta, 'f_hat': f_hat}


# ======= SVRG Algorithm for SVM ========

def SVRG(X, Y, C, learning_rate, batch_size=64, num_epochs=50):
    """Stochastic Variance Reduced Gradient (SVRG) optimizer."""
    if not isinstance(num_epochs, int) or num_epochs <= 0:
        raise ValueError("Must choose integer num_epochs > 0.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Must choose integer batch_size > 0.")
    
    n, d = X.shape
    X = np.hstack([np.ones((n, 1)), X])  # Adding intercept term
    M = int(np.ceil(n / batch_size))
    beta = np.zeros(d + 1)
    last_epoch_beta = np.zeros(d + 1)
    f_hat = np.zeros(num_epochs)
    
    for t in range(num_epochs):
        beta_old = last_epoch_beta.copy()
        
        full_grad_f = np.zeros(len(beta))
        for i in range(n):
            full_grad_f += svm_gradEval(X[i, :], Y[i], C, beta_old)
        
        ids = np.random.permutation(n)
        batch_ids = np.array_split(ids, M)
        
        for k in range(M):
            batch_X = X[batch_ids[k], :]
            batch_Y = Y[batch_ids[k]].reshape(-1, 1)
            
            checkpoint_grad_f = np.zeros(len(beta))
            for i in range(batch_X.shape[0]):
                checkpoint_grad_f += svm_gradEval(batch_X[i, :], batch_Y[i], C, beta_old)
            
            grad_f = np.zeros(len(beta))
            for i in range(batch_X.shape[0]):
                grad_f += svm_gradEval(batch_X[i, :], batch_Y[i], C, beta)
            
            beta -= learning_rate * (1 / batch_X.shape[0] * grad_f - 
                                     1 / batch_X.shape[0] * checkpoint_grad_f + 
                                     1 / n * full_grad_f)
        
        # Update last epoch beta
        last_epoch_beta = beta.copy()
        f_hat[t] = svm_objEval(X, Y, C, last_epoch_beta)
    
    return {'beta': last_epoch_beta, 'f_hat': f_hat}


# ======= ADMM Algorithm for Group LASSO =======
import numpy as np

def gl_beta_update(X_g, Y, alpha_g, z_g, rho):
    """Update the beta parameter for group g."""
    n, m_g = X_g.shape
    beta_next = np.linalg.solve(X_g.T @ X_g + rho * np.eye(m_g), (X_g.T @ Y / n + rho * alpha_g - z_g))
    return beta_next

def gl_alpha_update(X_g, beta_g, z_g, lambda_, rho):
    """Update the alpha parameter for group g."""
    m_g = X_g.shape[1]
    frac = lambda_ * np.sqrt(m_g) / (rho * np.linalg.norm(beta_g + z_g / rho, 2))
    alpha_next = max(0, 1 - frac) * (beta_g + z_g / rho)
    return alpha_next

def gl_z_update(z_g, beta_g, alpha_g, rho):
    """Update the z parameter."""
    z_next = z_g + rho * (beta_g - alpha_g)
    return z_next

def gl_lossEval(X, Y, groups, beta, lambda_):
    """Evaluate the Group LASSO objective function."""
    G = np.max(groups)
    n = X.shape[0]
    vector_sum = np.zeros(n)
    scalar_sum = 0
    
    for g in range(1, G+1):
        idx = np.where(groups == g)[0]
        X_g = X[:, idx]
        m_g = X_g.shape[1]
        beta_g = beta[idx]
        vector_sum += X_g @ beta_g
        scalar_sum += np.sqrt(m_g) * np.linalg.norm(beta_g, 2)
    
    loss = 1 / (2 * n) * np.linalg.norm(Y - vector_sum, 2)**2 + lambda_ * scalar_sum
    return loss

def groupLASSO(X, Y, groups, lambda_, rho=100, tol=1e-4, max_iter=10000, min_its=2):
    """Group LASSO using ADMM."""
    n, p = X.shape
    G = np.max(groups)
    
    # Standardize X and Y
    X_col_means = X.mean(axis=0)
    X_col_sd = X.std(axis=0, ddof=0)
    X_scaled = (X - X_col_means) / X_col_sd
    Y_mean = Y.mean()
    Y_scaled = Y - Y_mean

    # Initialize variables
    k = 0
    converged = False
    beta_scaled = np.zeros(p)
    alpha = np.zeros(p)
    z = np.zeros(p)
    loss = 0

    # ADMM loop
    while k <= max_iter and not converged:
        prev_loss = loss

        for g in range(1, G+1):
            idx = np.where(groups == g)[0]
            X_g = X_scaled[:, idx]

            # Update beta for group g
            beta_scaled[idx] = gl_beta_update(X_g, Y_scaled, alpha[idx], z[idx], rho)

            # Update alpha for group g
            alpha[idx] = gl_alpha_update(X_g, beta_scaled[idx], z[idx], lambda_, rho)

            # Update z for group g
            z[idx] = gl_z_update(z[idx], beta_scaled[idx], alpha[idx], rho)

        # Evaluate loss
        loss = gl_lossEval(X_scaled, Y_scaled, groups, beta_scaled, lambda_)
        diff = loss - prev_loss

        if k >= min_its and abs(diff) < tol:
            converged = True

        k += 1

    # Recover beta and intercept
    beta_scaled = alpha  # Final solution comes from alpha
    beta0 = Y_mean - np.sum(X_col_means * (beta_scaled / X_col_sd))
    beta = beta_scaled / X_col_sd

    return {'beta0': beta0, 'beta': beta}

